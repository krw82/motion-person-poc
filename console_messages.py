"""Compatibility entry point; implementation lives in motion_person.console_messages."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.console_messages")

sys.modules[__name__] = _implementation
