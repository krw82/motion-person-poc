"""JSONL 이벤트 로그, 실행 설정 기록과 종료 요약 (SPEC.md 20절, 21절).

실행별 폴더 log_dir/run_id 아래에 세 개 파일을 남긴다 (SPEC.md 20.1).

- run_config.json : 최종 유효 설정 + 패키지 버전 (versions)
- events.jsonl    : 공통 필드(SPEC.md 20.2)를 갖는 이벤트 스트림
- summary.json    : 종료 사유와 카운터 요약

직렬화 규칙은 UTF-8, ensure_ascii=False, allow_nan=False 이며 NaN 과
Infinity 를 기록하지 않는다. 로그 쓰기 실패는 재현성 손실이므로
LogWriteError 로 종료한다 (SPEC.md 20.6, 21절 LOG_WRITE_ERROR 행).
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts import LogWriteError

# 로그 규격 버전 (SPEC.md 20.2).
SCHEMA_VERSION = 1

_RUN_CONFIG_FILENAME = "run_config.json"
_EVENTS_FILENAME = "events.jsonl"
_SUMMARY_FILENAME = "summary.json"


def _utc_timestamp() -> str:
    """실제 UTC 시각을 마이크로초 정확도 ISO 문자열 + 'Z' 로 반환한다."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _distribution_version(names: tuple[str, ...]) -> str | None:
    """importlib.metadata 로 배포 버전을 찾는다 (best effort, 없으면 None)."""
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except OSError:
            # 메타데이터 조회 실패도 best effort 로 무시한다.
            continue
    return None


def _collect_versions() -> dict[str, str | None]:
    """run_config.json 에 병합할 패키지 버전 정보를 모은다.

    ultralytics 는 설치되지 않은 검증 환경에서 null 이 될 수 있다.
    """
    return {
        "python": platform.python_version(),
        "opencv": _distribution_version(("opencv-python", "opencv_python", "cv2")),
        "numpy": _distribution_version(("numpy",)),
        "ultralytics": _distribution_version(("ultralytics",)),
    }


class RunLogger:
    """실행별 로그 폴더와 JSONL/JSON 파일을 관리한다 (SPEC.md 20.6).

    사용 순서는 start -> write/write_summary -> close 다. close 는 멱등하고,
    모든 쓰기 실패는 LogWriteError 로 변환된다.
    """

    def __init__(self, log_dir: Path, run_id: str) -> None:
        self._log_dir = Path(log_dir)
        self._run_id = run_id
        self._run_dir = self._log_dir / run_id
        self._events_stream: Any = None

    # ------------------------------------------------------------------
    # 조회 전용 속성
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        """이 실행의 로그 폴더 (log_dir/run_id)."""
        return self._run_dir

    @property
    def events_path(self) -> Path:
        """events.jsonl 경로."""
        return self._run_dir / _EVENTS_FILENAME

    @property
    def config_path(self) -> Path:
        """run_config.json 경로."""
        return self._run_dir / _RUN_CONFIG_FILENAME

    @property
    def summary_path(self) -> Path:
        """summary.json 경로."""
        return self._run_dir / _SUMMARY_FILENAME

    # ------------------------------------------------------------------
    # 공개 인터페이스 (SPEC.md 20.6)
    # ------------------------------------------------------------------

    def start(self, config: dict, video_info: dict) -> None:
        """로그 폴더를 만들고 설정을 기록한 뒤 RUN_START 를 남긴다.

        - log_dir/run_id 생성. 이미 존재하면 LogWriteError.
        - run_config.json 작성: config 에 versions{python, opencv, numpy,
          ultralytics|null} 를 병합한다.
        - events.jsonl 을 열고 RUN_START 이벤트를 기록한다.
        """
        if self._events_stream is not None:
            raise LogWriteError(
                "RunLogger.start called twice",
                context={"run_id": self._run_id, "path": str(self._run_dir)},
            )

        try:
            self._run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise LogWriteError(
                f"log run dir already exists: {self._run_dir}",
                context={"run_id": self._run_id, "path": str(self._run_dir)},
            ) from exc
        except OSError as exc:
            raise LogWriteError(
                f"cannot create log run dir: {self._run_dir}",
                context={"run_id": self._run_id, "path": str(self._run_dir),
                         "reason": str(exc)},
            ) from exc

        merged_config = dict(config)
        merged_config["versions"] = _collect_versions()
        self._write_json_exclusive(self.config_path, merged_config, "run_config.json")

        try:
            self._events_stream = self.events_path.open("x", encoding="utf-8")
        except OSError as exc:
            self._events_stream = None
            raise LogWriteError(
                f"cannot open events file: {self.events_path}",
                context={"run_id": self._run_id, "path": str(self.events_path),
                         "reason": str(exc)},
            ) from exc

        # RUN_START 는 시작 설정과 환경 준비를 남긴다 (SPEC.md 20.3).
        self.write("RUN_START", {"config": merged_config, "video": video_info})

    def write(self, event_type: str, payload: dict) -> None:
        """공통 필드(SPEC.md 20.2)를 깔고 payload 로 덮어쓴 이벤트를 기록한다.

        공통 필드는 schema_version=1, event_type, timestamp_utc, run_id,
        frame_index=null, video_time_sec=null 이며 payload 가 우선한다.
        매 쓰기마다 flush 하고 개행으로 한 줄을 확정한다.
        """
        if self._events_stream is None:
            raise LogWriteError(
                "events.jsonl is not open; call start() first",
                context={"run_id": self._run_id},
            )

        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_type": event_type,
            "timestamp_utc": _utc_timestamp(),
            "run_id": self._run_id,
            "frame_index": None,
            "video_time_sec": None,
        }
        event.update(payload)

        try:
            line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            # allow_nan=False 의 NaN/Infinity 거부(ValueError)와 직렬화 불가
            # 값(TypeError) 모두 로그 쓰기 실패로 취급한다.
            raise LogWriteError(
                f"event serialization failed: {event_type}",
                context={"run_id": self._run_id, "event_type": event_type,
                         "reason": str(exc)},
            ) from exc

        try:
            self._events_stream.write(line + "\n")
            self._events_stream.flush()
        except (OSError, ValueError) as exc:
            raise LogWriteError(
                f"event write failed: {event_type}",
                context={"run_id": self._run_id, "event_type": event_type,
                         "path": str(self.events_path), "reason": str(exc)},
            ) from exc

    def write_summary(self, summary: dict) -> None:
        """summary.json 을 작성한다. 이미 존재하면 LogWriteError.

        직렬화 규칙은 events.jsonl 과 동일하다 (UTF-8, ensure_ascii=False,
        allow_nan=False).
        """
        if not self._run_dir.is_dir():
            raise LogWriteError(
                "run log dir missing; call start() first",
                context={"run_id": self._run_id, "path": str(self._run_dir)},
            )
        self._write_json_exclusive(self.summary_path, dict(summary), "summary.json")

    def close(self) -> None:
        """events.jsonl 을 flush 하고 닫는다. 중복 호출은 안전하다."""
        stream = self._events_stream
        self._events_stream = None
        if stream is None:
            return
        try:
            stream.flush()
            stream.close()
        except (OSError, ValueError) as exc:
            raise LogWriteError(
                "failed to close events file",
                context={"run_id": self._run_id,
                         "path": str(self.events_path), "reason": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _write_json_exclusive(self, path: Path, data: dict, label: str) -> None:
        """JSON 파일을 배타적('x') 모드로 쓴다. 충돌과 직렬화 실패를 막는다."""
        try:
            text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)
        except (TypeError, ValueError) as exc:
            raise LogWriteError(
                f"{label} serialization failed",
                context={"run_id": self._run_id, "path": str(path),
                         "reason": str(exc)},
            ) from exc
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
                stream.write("\n")
                stream.flush()
        except FileExistsError as exc:
            raise LogWriteError(
                f"{label} already exists: {path}",
                context={"run_id": self._run_id, "path": str(path)},
            ) from exc
        except OSError as exc:
            raise LogWriteError(
                f"cannot write {label}: {path}",
                context={"run_id": self._run_id, "path": str(path),
                         "reason": str(exc)},
            ) from exc
