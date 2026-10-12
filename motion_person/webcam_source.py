"""Compatibility import; edit motion_person.adapters.webcam instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.adapters.webcam")
sys.modules[__name__] = _implementation
