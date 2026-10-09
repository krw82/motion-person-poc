"""Compatibility entry point; implementation lives in motion_person.run_logger."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.run_logger")

sys.modules[__name__] = _implementation
