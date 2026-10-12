"""Compatibility import; edit motion_person.domain.photo_selection instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.domain.photo_selection")
sys.modules[__name__] = _implementation
