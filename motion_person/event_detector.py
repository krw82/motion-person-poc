"""Compatibility import; edit motion_person.domain.motion_rules instead."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.domain.motion_rules")
sys.modules[__name__] = _implementation
