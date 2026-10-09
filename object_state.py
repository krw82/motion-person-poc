"""Compatibility entry point; implementation lives in motion_person.object_state."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.object_state")

sys.modules[__name__] = _implementation
