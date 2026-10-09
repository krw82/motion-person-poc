"""Compatibility entry point; implementation lives in motion_person.capture_manager."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.capture_manager")

sys.modules[__name__] = _implementation
