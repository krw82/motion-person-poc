"""Compatibility entry point; implementation lives in motion_person.contracts."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.contracts")

sys.modules[__name__] = _implementation
