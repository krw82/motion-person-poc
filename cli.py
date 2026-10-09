"""Compatibility entry point; implementation lives in motion_person.cli."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.cli")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
