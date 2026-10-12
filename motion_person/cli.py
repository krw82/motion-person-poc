"""Compatibility import; edit motion_person.presentation.cli instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.presentation.cli")
if __name__ == "__main__":
    sys.exit(_implementation.main())
sys.modules[__name__] = _implementation
