"""Compatibility import; edit motion_person.application.event_capture instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.application.event_capture")
sys.modules[__name__] = _implementation
