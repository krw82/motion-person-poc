"""Compatibility import; edit motion_person.application.engine instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.application.engine")
sys.modules[__name__] = _implementation
