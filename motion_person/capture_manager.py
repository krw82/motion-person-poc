"""Compatibility import; edit motion_person.presentation.legacy.capture instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.presentation.legacy.capture")
sys.modules[__name__] = _implementation
