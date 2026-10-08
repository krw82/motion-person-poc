"""화면 표시 모듈 (SPEC.md 15.2, 17절).

analysis_frame.copy() 위에 사람/전경 박스와 상태 문구를 그려 반환하고
OpenCV 시연 창을 관리한다. 판정, 저장, 로그 기록은 하지 않는다.

주요 계약:
- render_overlay 는 packet.analysis_frame.copy() 를 반환용 캔버스로 쓰며
  호출자가 넘긴 배열은 수정하지 않는다 (SPEC.md 8.1, 11.3).
- 시연 화면 문구는 전부 영어다. OpenCV putText 기본 폰트는 한국어를
  지원하지 않는다 (SPEC.md 17.3).
- --no-display 실행에서는 DisplayWindow 의 어떤 메서드도 호출하지 않는다
  (SPEC.md 17.4).
- imshow/waitKey/창 생성/창 제거 오류는 DisplayError 로 보고한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from contracts import (
    Box,
    CaptureResult,
    ContractError,
    DisplayError,
    FrameDecision,
    FramePacket,
    MotionResult,
    PersonDetection,
)

__all__ = [
    "render_overlay",
    "render_mask_view",
    "DisplayWindow",
    "MAIN_WINDOW_NAME",
    "MASK_WINDOW_NAME",
]

# 창 이름 (SPEC.md 17.1)
MAIN_WINDOW_NAME = "Motion Person PoC"
MASK_WINDOW_NAME = "Valid Foreground Mask"

# 박스/문구 색상 (BGR, SPEC.md 17.2)
COLOR_PERSON = (0, 200, 0)      # YOLO 사람 박스: 녹색
COLOR_FG = (0, 0, 255)          # MOG2 전경 영역: 적색
COLOR_MATCH = (0, 255, 255)     # 적격 사람 강조: 노랑
COLOR_PENDING = (160, 160, 160)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_SHADOW = (0, 0, 0)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2
_MATCH_THICKNESS = 4

# metrics dict 에서 관찰할 수 있는 키 (SPEC.md 19.3, 20.5).
_METRIC_PROCESSING_FPS = "processing_fps"
_METRIC_PLAYBACK_FPS = "playback_fps"
_METRIC_CAPTURE_COUNT = "successful_capture_count"


# ---------------------------------------------------------------------------
# 텍스트/박스 그리기 헬퍼
# ---------------------------------------------------------------------------


def _text_metrics(text: str) -> tuple[int, int, int]:
    """putText 문구의 (너비, 글자 높이, 베이스라인)을 반환한다."""
    (width, height), baseline = cv2.getTextSize(
        text, _FONT, _FONT_SCALE, _FONT_THICKNESS
    )
    return int(width), int(height), int(baseline)


def _fit_text(text: str, max_width: int) -> str:
    """이미지보다 넓은 문구는 오른쪽부터 잘라 화면 안에 맞춘다."""
    if max_width <= 0:
        return ""
    while text and _text_metrics(text)[0] > max_width:
        text = text[:-1]
    return text


def _draw_text(
    frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]
) -> None:
    """그림자를 넣어 가독성을 높인 문구를 그린다. (x, y) 는 putText 기준점."""
    cv2.putText(
        frame, text, (x + 1, y + 1), _FONT, _FONT_SCALE,
        COLOR_TEXT_SHADOW, _FONT_THICKNESS, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (x, y), _FONT, _FONT_SCALE, color,
        _FONT_THICKNESS, cv2.LINE_AA,
    )


def _clip_text_origin(
    x: int, y: int, text_w: int, text_h: int, baseline: int,
    width: int, height: int,
) -> tuple[int, int]:
    """텍스트 기준점(bottom-left)을 이미지 안으로 clip 한다 (SPEC.md 17.3)."""
    org_x = int(max(4, min(x, max(4, width - text_w - 4))))
    org_y = int(max(2, min(y, max(2, height - 2))))
    # 여백이 되면 글자 상단까지 화면 안에 들어오게 기준점을 내린다.
    org_y = max(org_y, min(text_h + baseline + 2, max(2, height - 2)))
    return org_x, org_y


def _draw_box(
    frame: np.ndarray, box: Box, color: tuple[int, int, int],
    thickness: int, width: int, height: int,
) -> None:
    """박스를 화면 범위로 clip 해 그린다. x2/y2 는 제외 좌표다 (SPEC.md 8.1)."""
    x1 = max(0, min(box.x1, width))
    y1 = max(0, min(box.y1, height))
    x2 = max(0, min(box.x2, width))
    y2 = max(0, min(box.y2, height))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return  # clip 뒤 크기가 0이 되면 그리지 않는다
    cv2.rectangle(
        frame, (x1, y1), (x2 - 1, y2 - 1), color, thickness, cv2.LINE_AA
    )


def _draw_top_label(
    frame: np.ndarray, box: Box, text: str,
    color: tuple[int, int, int], width: int, height: int,
) -> None:
    """박스 왼쪽 위에 라벨을 그린다. 위가 화면 밖이면 박스 안쪽으로 넣는다."""
    text = _fit_text(text, max(1, width - 8))
    if not text:
        return
    text_w, text_h, baseline = _text_metrics(text)
    y = box.y1 - 6
    if y - text_h - baseline < 0:
        y = box.y1 + text_h + baseline + 4
    x, y = _clip_text_origin(box.x1, y, text_w, text_h, baseline, width, height)
    _draw_text(frame, text, x, y, color)


def _draw_match_label(
    frame: np.ndarray, box: Box, pixels: int, ratio: float,
    width: int, height: int,
) -> None:
    """적격 사람 박스 아래에 MATCH 라벨을 그린다 (형식은 SPEC.md 17.2)."""
    text = _fit_text(
        f"MATCH px={pixels} ratio={ratio:.3f}", max(1, width - 8)
    )
    if not text:
        return
    text_w, text_h, baseline = _text_metrics(text)
    y = box.y2 + text_h + baseline + 2
    if y > height - 2:
        y = box.y2 - 6
    x, y = _clip_text_origin(box.x1, y, text_w, text_h, baseline, width, height)
    _draw_text(frame, text, x, y, COLOR_MATCH)


# ---------------------------------------------------------------------------
# 상태 패널과 범례 (SPEC.md 17.3)
# ---------------------------------------------------------------------------


def _capture_label(capture: CaptureResult) -> str:
    """저장 상태 문구 (SPEC.md 15.2)."""
    if capture.status == "SAVED":
        return "CAPTURE SAVED"
    if capture.status == "COOLDOWN":
        remaining = max(0.0, float(capture.cooldown_remaining_sec))
        return f"COOLDOWN {remaining:.1f}s"
    if capture.status == "DISABLED":
        return "SAVING DISABLED"
    return "CAPTURE NONE"


def _fmt_float(value: object) -> str:
    """metrics 값이 관찰될 때만 소수 1자리로, 아니면 '-'."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if np.isfinite(value):
            return f"{float(value):.1f}"
    return "-"


def _fmt_int(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if np.isfinite(value):
            return str(int(value))
    return "-"


def _status_lines(
    packet: FramePacket,
    decision: FrameDecision,
    capture: CaptureResult,
    metrics: dict,
) -> list[str]:
    """화면 상태 문구 (SPEC.md 17.3 의 1~6 번 항목)."""
    qualified = [item for item in decision.persons if item.qualifies]
    if metrics.get("preview"):
        return [
            "WEBCAM PREVIEW | NO IMAGE SAVING",
            f"DETECTED {len(decision.objects)} MOVING {len(qualified)}",
            f"STATE {decision.status}",
            f"TIME {packet.video_time_sec:.1f}s FRAME {packet.frame_index}",
            "FPS {} AGE {} ms".format(
                _fmt_float(metrics.get(_METRIC_PROCESSING_FPS)),
                _fmt_float(metrics.get("frame_age_ms")),
            ),
            "GREEN detected | YELLOW motion",
            "GRAY pending | ESC / Q quit",
        ]

    # 4. 대표 적격 사람: 전경 픽셀 수가 가장 큰 사람.
    if qualified:
        best = max(qualified, key=lambda item: item.motion_pixels)
        best_label = (
            f"BEST MATCH px={best.motion_pixels} "
            f"ratio={best.motion_ratio:.3f}"
        )
    else:
        best_label = "BEST MATCH none"

    # 5/6. metrics 는 관찰된 키만 사용한다.
    total_captures = metrics.get(_METRIC_CAPTURE_COUNT)
    if total_captures is None and capture.sequence is not None:
        total_captures = capture.sequence

    return [
        f"STATE {decision.status} | {_capture_label(capture)}",
        f"TIME {packet.video_time_sec:.2f}s FRAME {packet.frame_index}",
        f"OBJECTS {len(decision.objects)} MATCHED {len(qualified)}",
        best_label,
        "FPS PROC {} PLAY {}".format(
            _fmt_float(metrics.get(_METRIC_PROCESSING_FPS)),
            _fmt_float(metrics.get(_METRIC_PLAYBACK_FPS)),
        ),
        f"CAPTURES {_fmt_int(total_captures)}",
    ]


def _draw_status_panel(
    frame: np.ndarray, lines: list[str], width: int, height: int
) -> None:
    """좌상단에 검정 배경 상태 패널을 그린다."""
    _, text_h, baseline = _text_metrics("Ag")
    line_h = text_h + baseline + 6
    panel_w = min(
        width,
        max((_text_metrics(line)[0] for line in lines), default=0) + 12,
    )
    panel_h = min(height, line_h * len(lines) + 6)
    cv2.rectangle(
        frame, (0, 0), (panel_w, panel_h), COLOR_TEXT_SHADOW, thickness=-1
    )
    for index, line in enumerate(lines):
        y = 3 + text_h + index * line_h
        if y > height - 2:
            break
        text = _fit_text(line, max(1, width - 12))
        if not text:
            continue
        cv2.putText(
            frame, text, (6, y), _FONT, _FONT_SCALE, COLOR_TEXT,
            _FONT_THICKNESS, cv2.LINE_AA,
        )


def _draw_legend(frame: np.ndarray, width: int, height: int) -> None:
    """좌하단에 색상 범례를 작게 그린다 (SPEC.md 17.2)."""
    entries = (
        ("OBJECT", COLOR_PERSON),
        ("FG", COLOR_FG),
        ("MATCH", COLOR_MATCH),
    )
    square = 9
    pad = 6
    gap = 14
    x = pad
    for name, color in entries:
        text_w, _, _ = _text_metrics(name)
        needed = square + 4 + text_w
        if x + needed > width - 2:
            break
        top = height - pad - square
        if top < 0:
            break
        cv2.rectangle(
            frame, (x, top), (x + square, top + square), color, thickness=-1
        )
        _draw_text(frame, name, x + square + 4, height - pad, color)
        x += needed + gap


# ---------------------------------------------------------------------------
# 공개 함수 (SPEC.md 17.1)
# ---------------------------------------------------------------------------


def render_overlay(
    packet: FramePacket,
    persons: tuple[PersonDetection, ...],
    motion: MotionResult,
    decision: FrameDecision,
    capture: CaptureResult,
    metrics: dict,
) -> np.ndarray:
    """분석 결과를 그린 display 프레임을 반환한다 (SPEC.md 17.1~17.3).

    반환값은 packet.analysis_frame.copy() 위에 박스/라벨/상태를 그린
    새 배열이다. 호출자가 넘긴 프레임과 마스크는 수정하지 않는다.
    """
    analysis = packet.analysis_frame
    if (
        not isinstance(analysis, np.ndarray)
        or analysis.ndim != 3
        or analysis.dtype != np.uint8
    ):
        raise ContractError(
            "analysis_frame must be HxWx3 uint8 BGR",
            context={"frame_index": packet.frame_index},
        )

    display = analysis.copy()  # SPEC.md 11.3
    height, width = display.shape[:2]

    # 1) MOG2 전경 영역: 적색 FG (SPEC.md 17.2).
    for region in motion.regions:
        _draw_box(display, region.box, COLOR_FG, _BOX_THICKNESS, width, height)
        _draw_top_label(display, region.box, "FG", COLOR_FG, width, height)

    # 2) YOLO 사람 박스: 녹색 PERSON <conf>.
    for person in persons:
        color = COLOR_PERSON if person.confirmed else COLOR_PENDING
        _draw_box(
            display, person.box, color, _BOX_THICKNESS, width, height
        )
        confirmation = "" if person.confirmed else f" PENDING hits={person.observation_hits}"
        _draw_top_label(
            display, person.box,
            f"{person.label.upper()} {person.confidence:.2f}{confirmation}", color, width, height,
        )

    # 3) 적격 사람: 노랑 두께 증가 박스 + MATCH 라벨 (decision.persons 의
    #    qualifies 로 매칭, px/ratio 는 실측값을 반올림해 표시).
    for evidence in decision.persons:
        if not evidence.qualifies:
            continue
        _draw_box(
            display, evidence.box, COLOR_MATCH,
            _MATCH_THICKNESS, width, height,
        )
        _draw_match_label(
            display, evidence.box, evidence.motion_pixels,
            evidence.motion_ratio, width, height,
        )

    # 4) 상태 패널과 색상 범례 (SPEC.md 17.3).
    if metrics is None:  # 방어: main 은 항상 dict 를 넘기지만 None 도 허용
        metrics = {}
    if metrics.get("tracking") or metrics.get("preview"):
        # Keep the ID labels and small source frames visible beside the status panel.
        sidebar = np.zeros((max(height, 210), 380, 3), dtype=np.uint8)
        _draw_status_panel(sidebar, _status_lines(packet, decision, capture, metrics),
                           sidebar.shape[1], sidebar.shape[0])
        _draw_legend(sidebar, sidebar.shape[1], sidebar.shape[0])
        canvas = np.zeros((sidebar.shape[0], width + sidebar.shape[1], 3), dtype=np.uint8)
        canvas[:height, :width] = display
        canvas[:, width:] = sidebar
        return canvas
    _draw_status_panel(
        display, _status_lines(packet, decision, capture, metrics), width, height
    )
    _draw_legend(display, width, height)

    return display


def render_mask_view(motion: MotionResult) -> np.ndarray:
    """valid_mask 를 3채널 BGR 시각화로 변환한다 (--show-mask 용, SPEC.md 17.1).

    검정 배경에 유효 전경(255)만 흰색으로 표시한다. 입력 마스크는
    수정하지 않고 새 배열을 반환한다.
    """
    mask = motion.valid_mask
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ContractError(
            "valid_mask must be 2D uint8 with values 0/255",
            context={
                "shape": tuple(mask.shape) if isinstance(mask, np.ndarray) else None,
                "dtype": str(mask.dtype) if isinstance(mask, np.ndarray) else type(mask).__name__,
            },
        )
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# 시연 창 (SPEC.md 17.1, 17.4)
# ---------------------------------------------------------------------------


class DisplayWindow:
    """OpenCV 시연 창 관리.

    - 기본 창 이름은 Motion Person PoC, show_mask 를 켜면 Valid
      Foreground Mask 창을 추가로 만든다.
    - poll_key: ESC(27) -> "esc", q/Q -> "quit", 그 외/무입력 -> None.
    - is_closed: 창 닫기 버튼(WND_PROP_VISIBLE < 1)으로 닫혔는지 확인.
    - close: 멱등. 창 생성/표시/키 조회/제거 오류는 DisplayError.
    """

    def __init__(self, show_mask: bool = False) -> None:
        self._show_mask = bool(show_mask)
        self._mask_name: str | None = (
            MASK_WINDOW_NAME if self._show_mask else None
        )
        self._open = False
        try:
            cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            if self._mask_name is not None:
                cv2.namedWindow(self._mask_name, cv2.WINDOW_AUTOSIZE)
        except (cv2.error, TypeError) as exc:
            raise DisplayError(
                "failed to create display window",
                context={"window": MAIN_WINDOW_NAME, "reason": str(exc)},
            ) from exc
        self._open = True

    @property
    def show_mask(self) -> bool:
        """마스크 창이 활성화되어 있는지 여부."""
        return self._show_mask

    def show(
        self, display_frame: np.ndarray, mask_view: np.ndarray | None = None
    ) -> None:
        """현재 프레임을 창에 표시한다. 마스크 창이 있으면 mask_view 필수."""
        if not self._open:
            raise DisplayError(
                "display window is already closed",
                context={"window": MAIN_WINDOW_NAME},
            )
        if self._mask_name is not None and mask_view is None:
            raise DisplayError(
                "mask_view is required when the mask window is enabled",
                context={"window": self._mask_name},
            )
        try:
            cv2.imshow(MAIN_WINDOW_NAME, display_frame)
            if self._mask_name is not None:
                cv2.imshow(self._mask_name, mask_view)
        except (cv2.error, TypeError) as exc:
            raise DisplayError(
                "failed to show frame",
                context={"window": MAIN_WINDOW_NAME, "reason": str(exc)},
            ) from exc

    def poll_key(self) -> str | None:
        """키를 조회한다. ESC 는 "esc", q 는 "quit", 없으면 None."""
        if not self._open:
            return None
        try:
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            raise DisplayError(
                "waitKey failed", context={"reason": str(exc)}
            ) from exc
        if key == 27:  # ESC
            return "esc"
        if key in (ord("q"), ord("Q")):
            return "quit"
        return None

    def is_closed(self) -> bool:
        """사용자가 창 닫기 버튼으로 닫았는지 확인한다 (USER_STOP 판단)."""
        if not self._open:
            return True
        try:
            if cv2.getWindowProperty(
                MAIN_WINDOW_NAME, cv2.WND_PROP_VISIBLE
            ) < 1:
                return True
            if self._mask_name is not None:
                if cv2.getWindowProperty(
                    self._mask_name, cv2.WND_PROP_VISIBLE
                ) < 1:
                    return True
        except cv2.error:
            # 조회 대상 창이 이미 없으면 닫힌 것으로 본다.
            return True
        return False

    def close(self) -> None:
        """모든 창을 제거한다. 여러 번 호출해도 안전하다 (SPEC.md 10.5)."""
        if not self._open:
            return
        self._open = False
        try:
            cv2.destroyWindow(MAIN_WINDOW_NAME)
            if self._mask_name is not None:
                cv2.destroyWindow(self._mask_name)
        except cv2.error as exc:
            raise DisplayError(
                "failed to destroy display window",
                context={"window": MAIN_WINDOW_NAME, "reason": str(exc)},
            ) from exc
