"""Compatibility entry point; implementation lives in motion_person.overlay_renderer."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.overlay_renderer")

sys.modules[__name__] = _implementation
