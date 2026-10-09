"""Compatibility entry point; implementation lives in motion_person.motion_detector."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.motion_detector")

sys.modules[__name__] = _implementation
