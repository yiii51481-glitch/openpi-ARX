#!/usr/bin/env python3
"""Compatibility wrapper for the pick-bread ARX5 policy server."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.arx5.serve_policy import PROFILES
from scripts.arx5.serve_policy import main
from scripts.arx5.serve_policy import serve_profile

EXPECTED_CHECKPOINT_STEP = PROFILES["pick_bread"].expected_step


if __name__ == "__main__":
    main("pick_bread")
