"""Compatibility entry point; implementation lives in motion_person.frame_processor."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.frame_processor")

sys.modules[__name__] = _implementation
