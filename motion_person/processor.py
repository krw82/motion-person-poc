"""Compatibility import; edit motion_person.application.processor instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.application.processor")
sys.modules[__name__] = _implementation
