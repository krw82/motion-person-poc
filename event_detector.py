"""Compatibility entry point; implementation lives in motion_person.event_detector."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.event_detector")

sys.modules[__name__] = _implementation
