"""Compatibility import; edit motion_person.presentation.legacy.main instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.presentation.legacy.main")
if __name__ == "__main__":
    sys.exit(_implementation.main())
sys.modules[__name__] = _implementation
