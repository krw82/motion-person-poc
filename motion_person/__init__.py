"""Fixed-camera object motion capture, without cloud analysis."""
__version__ = "0.2.0"

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from .contracts import MotionPersonError
    from .engine import MotionEngine, RunSummary
    from .event_capture import EventBundle, EventConfig, EventImage, EventObject

_EXPORTS = {
    "MotionEngine": ".engine", "RunSummary": ".engine",
    "EventConfig": ".event_capture", "EventBundle": ".event_capture",
    "EventImage": ".event_capture", "EventObject": ".event_capture",
    "Config": ".config", "MotionPersonError": ".contracts",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    # Help/doctor must still run when optional runtime dependencies are missing.
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
