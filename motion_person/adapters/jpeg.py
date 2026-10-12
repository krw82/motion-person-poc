"""JPEG encoding boundary shared by evidence storage."""

import cv2
from motion_person.domain.contracts import CaptureSaveError


def encode_jpeg(frame, quality: int) -> bytes:
    try:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("JPEG encoder returned false")
        return encoded.tobytes()
    except (cv2.error, ValueError, TypeError) as exc:
        raise CaptureSaveError(
            "event JPEG encoding failed", context={"reason": str(exc)}
        ) from exc
