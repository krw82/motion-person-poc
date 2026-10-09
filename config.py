"""Compatibility entry point; implementation lives in motion_person.config."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.config")

sys.modules[__name__] = _implementation
