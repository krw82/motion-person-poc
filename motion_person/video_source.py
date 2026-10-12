"""Compatibility import; edit motion_person.adapters.video instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.adapters.video")
sys.modules[__name__] = _implementation
