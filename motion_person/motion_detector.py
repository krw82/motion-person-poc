"""Compatibility import; edit motion_person.adapters.motion instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.adapters.motion")
sys.modules[__name__] = _implementation
