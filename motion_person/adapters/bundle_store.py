"""Publish complete photo bundles using a staging directory."""

from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
from typing import Sequence
import uuid
from motion_person.domain.contracts import CaptureSaveError
from motion_person.domain.events import EventBundle


class FileBundleStore:
    def publish(self, bundle: EventBundle, images: Sequence[bytes]) -> None:
        final = bundle.manifest_path.parent
        staging = final.parent / f".{final.name}.{uuid.uuid4().hex}.tmp"
        published = False
        try:
            if len(images) != len(bundle.images):
                raise ValueError("photo payloads and metadata must align")
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                raise FileExistsError(f"bundle directory already exists: {final}")
            staging.mkdir()
            for image, jpeg in zip(bundle.images, images):
                self._write(staging / image.path.name, jpeg)
            manifest = json.dumps(
                bundle.to_dict(), ensure_ascii=False, allow_nan=False, indent=2
            )
            self._write(staging / "event.json", (manifest + "\n").encode("utf-8"))
            staging.rename(final)
            published = True
        except (OSError, ValueError, TypeError) as exc:
            raise CaptureSaveError(
                "event bundle publication failed",
                context={"path": str(final), "reason": str(exc)},
            ) from exc
        finally:
            if not published:
                shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
