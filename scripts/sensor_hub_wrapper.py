#!/usr/bin/env python3
"""Compatibility wrapper — canonical hub lives in the capstone repo."""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path("/home/siddartha/capstone/soccer_analytics/sensors/hub.py")),
    run_name="__main__",
)
