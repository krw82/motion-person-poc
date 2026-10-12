"""Fixed-camera object motion capture, without cloud analysis."""

__version__ = "0.2.0"

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .domain.settings import Config as Config
    from .domain.contracts import (
        FrameAnalysis as FrameAnalysis,
        MotionPersonError as MotionPersonError,
    )
    from .application.engine import (
        MotionEngine as MotionEngine,
        RunSummary as RunSummary,
    )
    from .domain.events import (
        EventBundle as EventBundle,
        EventConfig as EventConfig,
        EventImage as EventImage,
        EventObject as EventObject,
    )
    from .adapters.preview_encoder import (
        FramePreview as FramePreview,
        PreviewConfig as PreviewConfig,
    )

_EXPORTS = {
    "MotionEngine": ".application.engine",
    "RunSummary": ".application.engine",
    "EventConfig": ".domain.events",
    "EventBundle": ".domain.events",
    "EventImage": ".domain.events",
    "EventObject": ".domain.events",
    "Config": ".domain.settings",
    "MotionPersonError": ".domain.contracts",
    "FrameAnalysis": ".domain.contracts",
    "FramePreview": ".adapters.preview_encoder",
    "PreviewConfig": ".adapters.preview_encoder",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    # Help/doctor must still run when optional runtime dependencies are missing.
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
