"""Compatibility entry point; implementation lives in motion_person.tracking_adapter."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.tracking_adapter")

sys.modules[__name__] = _implementation
