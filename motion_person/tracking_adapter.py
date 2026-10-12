"""Compatibility import; edit motion_person.adapters.tracking instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.adapters.tracking")
sys.modules[__name__] = _implementation
