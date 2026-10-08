"""선택한 YOLO 객체 종류 탐지와 선택적 실행 내 추적.

기본값은 person이며 --classes와 --track으로 다중 종류 및 추적을 활성화한다.

설계 요점:
- ``from ultralytics import YOLO`` 는 load() 내부에서만 수행한다.
  top-level import 를 쓰면 ultralytics 미설치 환경에서
  ``import object_detector`` 자체가 실패한다.
- load 는 실행 시작 때 한 번만 호출한다 (FR04). 프레임 반복문 안에서
  모델을 생성하지 않는다.
- predict 예외를 사람 없음으로 위장하지 않고 InferenceError 로 감싼다
  (SPEC.md 12.5, 21절 INFERENCE_ERROR).
- detect는 입력을 수정하지 않고 프레임당 1회 추론한다. 추적 모드에서는
  persist=True로 추적 상태를 유지한다. 더미 warmup은 추적 상태를 변경하지 않는다.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any
from pathlib import Path

import numpy as np

from config import Config
from contracts import (
    Box,
    ContractError,
    InferenceError,
    ModelClassError,
    ModelLoadError,
    ModelNotFoundError,
    PersonDetection,
)

# sha256 계산 시 한 번에 읽을 청크 크기 (NFR03: 전체 파일을 메모리에 올리지 않음).
_HASH_CHUNK_BYTES = 1024 * 1024


class PersonDetector:
    """YOLO person 검출기 (SPEC.md 12.2 인터페이스).

    Attributes:
        model_sha256: load 시 계산한 가중치 파일의 sha256 hex (NFR02).
        person_class_id: 모델 names 에서 확인한 person 클래스 정수 id.
        invalid_detection_count: §12.5 정리 과정에서 제외된 비정상 박스
            수의 누적값. 좌표/신뢰도 비유한, 클래스 불일치, 신뢰도 범위
            이탈, clip 후 폭·높이 0 을 모두 포함한다.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model: Any | None = None
        self._loaded = False
        self.model_sha256: str = ""
        self.person_class_id: int = -1
        self.class_names: dict[int, str] = {}
        self.target_class_ids: tuple[int, ...] = ()
        self.invalid_detection_count: int = 0
        self.untracked_detection_count: int = 0
        self._identity_ids: dict[tuple[int, int], int] = {}
        self._tracker_classes: dict[int, set[int]] = {}
        self.class_identity_splits: int = 0
        self._observation_hits: dict[int, int] = {}
        self._tracker_path: str | None = None

    @property
    def loaded(self) -> bool:
        """load() 완료 여부. detect/warmup 는 로드 뒤에만 유효하다."""
        return self._loaded

    # ------------------------------------------------------------------
    # 로드 (SPEC.md 12.3, 12.6)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """모델 가중치를 로드하고 person 클래스 매핑을 검증한다.

        절차:
        1. 모델 파일을 일반 파일로 다시 확인한다. 없으면 ModelNotFoundError.
        2. sha256 을 청크로 읽어 계산해 self.model_sha256 에 노출한다.
        3. YOLO(str(path)) 생성에 실패하면 ModelLoadError.
        4. names 에서 이름이 person 인 id 를 수집한다. 0개이거나 2개
           이상이면 ModelClassError (§12.3: 이번 계약에 맞지 않는 모델).
        """
        if self._loaded:
            return

        path = self.config.model_path
        if not path.is_file():
            raise ModelNotFoundError(
                f"model file not found: {path}",
                context={"path": str(path), "code": "MODEL_NOT_FOUND"},
            )
        self.model_sha256 = _hash_file(path)

        try:
            from ultralytics import YOLO  # lazy import: load() 안에서만 허용
        except Exception as exc:
            raise ModelLoadError(
                "ultralytics package import failed; check requirements",
                context={"path": str(path), "reason": str(exc)},
            ) from exc

        try:
            model = YOLO(str(path))
        except Exception as exc:
            raise ModelLoadError(
                f"failed to load YOLO weights: {path}",
                context={"path": str(path), "reason": str(exc)},
            ) from exc

        self.class_names, self.target_class_ids = resolve_target_classes(
            getattr(model, "names", None), self.config.target_classes
        )
        person_ids = _find_person_class_ids(getattr(model, "names", None))
        self._model = model
        self.person_class_id = int(person_ids[0]) if len(person_ids) == 1 else -1
        self._loaded = True
        if self.config.tracking and self.config.tracking_profile == "stable":
            self._model.add_callback("on_predict_start", self._start_class_tracking)

    def prepare_tracking(self, path: Path) -> dict:
        """Persist the exact effective tracker YAML in this run's log folder."""
        from ultralytics.utils import YAML, ROOT

        settings = YAML.load(ROOT / "cfg/trackers" / f"{self.config.tracker}.yaml")
        settings.update(track_buffer=self.config.track_buffer,
                        new_track_thresh=self.config.track_new_threshold)
        if self.config.tracker == "botsort":
            settings.update(with_reid=self.config.track_reid, model="auto")
            if self.config.tracking_profile == "stable":
                settings["gmc_method"] = "none"  # user deployment: fixed cameras
        YAML.save(path, settings)
        self._tracker_path = str(path.resolve())
        return settings

    def _start_class_tracking(self, predictor) -> None:
        if predictor.args.mode != "track":
            return  # dummy model warmup never advances or creates tracks
        from tracking_adapter import ClassSeparatedTracker
        from ultralytics.trackers.track import on_predict_start

        if (hasattr(predictor, "trackers") and predictor.trackers
                and isinstance(predictor.trackers[0], ClassSeparatedTracker)):
            return
        # Use the official feature hook setup, then replace only tracker state.
        on_predict_start(predictor, persist=False)
        predictor.trackers = [ClassSeparatedTracker(type(t), t.args, self.target_class_ids)
                              for t in predictor.trackers]

    # ------------------------------------------------------------------
    # 추론 (SPEC.md 12.4, 12.5)
    # ------------------------------------------------------------------

    def detect(self, analysis_frame: np.ndarray) -> tuple[PersonDetection, ...]:
        """분석 프레임에서 person 박스를 PersonDetection 으로 반환한다.

        - predict 호출 인자는 §12.4 예시를 따른다 (classes, conf, iou,
          imgsz, device=cpu, max_det, verbose=False, save=False).
        - 사람이 없으면 빈 튜플을 반환한다.
        - §12.5 정리: 유한성 확인, x1/y1 floor 와 x2/y2 ceil, [0,w]/[0,h]
          clip, 폭·높이 0 제외, (-conf, x1, y1, x2, y2) 정렬 후
          detection_index 를 순차 부여한다.
        - load 전 호출은 ContractError, 추론 실패는 InferenceError.
        - 입력 프레임은 수정하지 않는다.
        """
        if not self._loaded or self._model is None:
            raise ContractError(
                "PersonDetector.detect called before load()",
                context={"model_path": str(self.config.model_path)},
            )
        width, height = _validate_analysis_frame(analysis_frame)
        shape_context = {"shape": [int(height), int(width), 3]}

        try:
            kwargs = dict(
                source=analysis_frame,
                classes=list(self.target_class_ids),
                conf=self.config.person_confidence,
                iou=self.config.yolo_iou,
                imgsz=self.config.yolo_imgsz,
                device="cpu",
                max_det=self.config.max_detections,
                verbose=False,
                save=False,
            )
            if self.config.tracking:
                # Keep weak detections available to the tracker's association stage.
                # Exposed detections and capture decisions still use the user's confidence gate.
                kwargs["conf"] = min(self.config.person_confidence, 0.1)
                results = self._model.track(
                    **kwargs, persist=True, tracker=self._tracker_path or f"{self.config.tracker}.yaml"
                )
            else:
                results = self._model.predict(**kwargs)
        except Exception as exc:
            raise InferenceError(
                "YOLO predict failed",
                context={**shape_context, "reason": str(exc),
                         "exception_type": type(exc).__name__},
            ) from exc

        try:
            result_count = len(results)
        except TypeError as exc:
            raise InferenceError(
                "YOLO predict returned an unexpected result container",
                context={**shape_context, "result_type": type(results).__name__},
            ) from exc
        if result_count != 1:
            # §12.4: 결과가 한 프레임 입력과 대응하는지 확인한다.
            raise InferenceError(
                "YOLO result count does not match single-frame input",
                context={**shape_context, "result_count": int(result_count)},
            )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return ()

        try:
            coordinates = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id
            track_ids = ids.cpu().numpy() if ids is not None else None
        except Exception as exc:
            raise InferenceError(
                "failed to read YOLO box tensors",
                context={**shape_context, "reason": str(exc),
                         "exception_type": type(exc).__name__},
            ) from exc

        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise InferenceError(
                "YOLO xyxy tensor has unexpected shape",
                context={**shape_context, "xyxy_shape": list(coordinates.shape)},
            )
        if not (len(coordinates) == len(confidences) == len(classes)):
            raise InferenceError(
                "YOLO box tensor lengths are inconsistent",
                context={**shape_context,
                         "xyxy": int(len(coordinates)),
                         "conf": int(len(confidences)),
                         "cls": int(len(classes))},
            )
        count = len(coordinates)
        if (confidences.shape != (count,) or classes.shape != (count,)
                or (track_ids is not None and track_ids.shape != (count,))):
            raise InferenceError("YOLO result fields do not align", context=shape_context)

        if self.config.tracking and track_ids is None:
            # Unconfirmed detections are not assigned invented identities.
            self.untracked_detection_count += count
            return ()

        return self._cleanup_detections(coordinates, confidences, classes,
                                        width, height, track_ids)

    def _cleanup_detections(
        self,
        coordinates: np.ndarray,
        confidences: np.ndarray,
        classes: np.ndarray,
        width: int,
        height: int,
        track_ids: np.ndarray | None = None,
    ) -> tuple[PersonDetection, ...]:
        """§12.5 박스 정리: 검증, 정수화, clip, 제외, 정렬, 번호 부여."""
        candidates = []
        seen_ids: set[int] = set()
        min_conf = self.config.person_confidence

        for index in range(len(coordinates)):
            row = coordinates[index]
            confidence = float(confidences[index])

            # 1. 좌표와 confidence 가 유한한 수인지 확인한다.
            if not math.isfinite(confidence) or not bool(np.all(np.isfinite(row))):
                self.invalid_detection_count += 1
                continue

            # 2. person ID 와 confidence 를 다시 검증한다.
            class_id = int(classes[index])
            if class_id not in self.target_class_ids:
                self.invalid_detection_count += 1
                continue

            track_id = None
            if track_ids is not None:
                raw_id = float(track_ids[index])
                if not math.isfinite(raw_id) or raw_id < 1 or raw_id != int(raw_id):
                    self.invalid_detection_count += 1
                    continue
                track_id = int(raw_id)
            if confidence < min_conf or confidence > 1.0:
                self.invalid_detection_count += 1
                continue

            # 3. x1, y1 은 floor, x2, y2 는 ceil.
            x1 = min(width, max(0, math.floor(float(row[0]))))
            y1 = min(height, max(0, math.floor(float(row[1]))))
            x2 = min(width, max(0, math.ceil(float(row[2]))))
            y2 = min(height, max(0, math.ceil(float(row[3]))))

            # 4. [0, w], [0, h] clip 적용 뒤 폭·높이가 0 이면 제외한다.
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                self.invalid_detection_count += 1
                continue

            candidates.append((confidence, x1, y1, x2, y2, class_id, track_id))
            if track_id is not None:
                if track_id in seen_ids:
                    raise InferenceError("tracker returned duplicate IDs in one frame",
                                         context={"track_id": track_id})
                seen_ids.add(track_id)

        # 5. confidence 내림차순, 좌표 오름차순 정렬 후 순차 번호를 부여한다.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
        cleaned = tuple(
            PersonDetection(
                detection_index=rank,
                box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=confidence,
                class_id=class_id,
                class_name=self.class_names[class_id],
                track_id=self._identity_id(class_id, track_id),
                tracker_id=track_id,
            )
            for rank, (confidence, x1, y1, x2, y2, class_id, track_id) in enumerate(candidates)
        )
        if not self.config.tracking:
            return cleaned
        from dataclasses import replace
        confirmed = []
        for detection in cleaned:
            track_id = detection.track_id
            hits = self._observation_hits.get(track_id, 0) + 1
            self._observation_hits[track_id] = hits
            confirmed.append(replace(detection, observation_hits=hits,
                                     confirmed=hits >= self.config.track_min_hits))
        return tuple(confirmed)

    def _identity_id(self, class_id: int, tracker_id: int | None) -> int | None:
        if tracker_id is None:
            return None
        key = (class_id, tracker_id)
        if key not in self._identity_ids:
            classes = self._tracker_classes.setdefault(tracker_id, set())
            if classes and class_id not in classes:
                self.class_identity_splits += 1
            classes.add(class_id)
            self._identity_ids[key] = len(self._identity_ids) + 1
        return self._identity_ids[key]

    # ------------------------------------------------------------------
    # 첫 추론 준비 (SPEC.md 12.6)
    # ------------------------------------------------------------------

    def warmup(self, width: int, height: int) -> float:
        """0 배열 더미 프레임으로 첫 추론을 1회 수행하고 경과 ms 를 반환한다.

        더미 프레임은 MOG2 에 전달하지 않는다(호출자 계약, §12.6).
        측정값은 model_init_ms 기록에 사용하며 steady state FPS 계산에서
        제외한다.
        """
        if not self._loaded or self._model is None:
            raise ContractError(
                "PersonDetector.warmup called before load()",
                context={"model_path": str(self.config.model_path)},
            )
        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContractError(
                    "warmup requires positive integer dimensions",
                    context={"width": repr(width), "height": repr(height)},
                )

        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        started = time.perf_counter()
        try:
            self._model.predict(
                source=dummy_frame, classes=list(self.target_class_ids),
                conf=self.config.person_confidence, iou=self.config.yolo_iou,
                imgsz=self.config.yolo_imgsz, device="cpu",
                max_det=self.config.max_detections, verbose=False, save=False,
            )
        except Exception as exc:
            raise InferenceError("YOLO warmup failed", context={"reason": str(exc)}) from exc
        return (time.perf_counter() - started) * 1000.0


def resolve_target_classes(names: Any, targets: tuple[str, ...]) -> tuple[dict[int, str], tuple[int, ...]]:
    if isinstance(names, dict):
        entries = names.items()
    elif isinstance(names, (list, tuple)):
        entries = enumerate(names)
    else:
        raise ModelClassError("model names must be a dictionary or list")
    mapping = {int(i): str(name).strip().lower() for i, name in entries}
    ids = []
    for target in targets:
        matches = [i for i, name in mapping.items() if name == target]
        if len(matches) != 1:
            raise ModelClassError(
                f"model must contain exactly one class named '{target}'",
                context={"class_name": target, "matches": matches,
                         "available_classes": sorted(mapping.values())},
            )
        ids.append(matches[0])
    return mapping, tuple(ids)


ObjectDetector = PersonDetector


# ----------------------------------------------------------------------
# 내부 헬퍼
# ----------------------------------------------------------------------


def _hash_file(path, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """파일을 청크 단위로 읽어 sha256 hex 다이제스트를 반환한다."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        # 읽기 권한 등 가중치 파일 접근 실패는 로드 실패로 보고한다(§21절).
        raise ModelLoadError(
            f"cannot read model file: {path}",
            context={"path": str(path), "reason": repr(exc)},
        ) from exc
    return digest.hexdigest()


def _find_person_class_ids(names: Any) -> list[int]:
    """모델 names 에서 이름이 'person' 인 클래스 id 목록을 반환한다.

    names 는 ultralytics 버전에 따라 dict{id: name} 또는 list[name] 형태다.
    """
    person_ids: list[int] = []
    if isinstance(names, dict):
        for key, value in names.items():
            if str(value) == "person":
                person_ids.append(int(key))
    elif isinstance(names, (list, tuple)):
        for index, value in enumerate(names):
            if str(value) == "person":
                person_ids.append(int(index))
    return person_ids


def _validate_analysis_frame(analysis_frame: np.ndarray) -> tuple[int, int]:
    """분석 프레임이 h x w x 3 uint8 계약(SPEC.md 8.1)을 지키는지 확인한다.

    반환값은 (width, height)다. 위반이면 ContractError.
    """
    if not isinstance(analysis_frame, np.ndarray):
        raise ContractError(
            "analysis_frame must be a numpy ndarray",
            context={"actual_type": type(analysis_frame).__name__},
        )
    if analysis_frame.ndim != 3 or analysis_frame.shape[2] != 3:
        raise ContractError(
            "analysis_frame must have shape H x W x 3",
            context={"shape": [int(dim) for dim in analysis_frame.shape]},
        )
    height, width = int(analysis_frame.shape[0]), int(analysis_frame.shape[1])
    if width <= 0 or height <= 0:
        raise ContractError(
            "analysis_frame dimensions must be positive",
            context={"width": width, "height": height},
        )
    if analysis_frame.dtype != np.uint8:
        raise ContractError(
            "analysis_frame must be uint8 (BGR)",
            context={"dtype": str(analysis_frame.dtype)},
        )
    return width, height
