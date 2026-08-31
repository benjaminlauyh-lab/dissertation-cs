#!/usr/bin/env python3
"""Convenience wrapper: run the dissertation lab in pilot mode (5 subjects)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_experiment  # noqa: E402


def main() -> None:
    argv = list(sys.argv[1:])
    if "--mode" not in argv:
        argv = ["--mode", "pilot", *argv]
    run_experiment.run(argv)


if __name__ == "__main__":
    main()
