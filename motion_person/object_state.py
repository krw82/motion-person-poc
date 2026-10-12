"""Compatibility import; edit motion_person.presentation.legacy.object_state instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.presentation.legacy.object_state")
sys.modules[__name__] = _implementation
