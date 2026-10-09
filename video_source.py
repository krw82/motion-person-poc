"""Compatibility entry point; implementation lives in motion_person.video_source."""
from importlib import import_module
import sys

_implementation = import_module("motion_person.video_source")

sys.modules[__name__] = _implementation
