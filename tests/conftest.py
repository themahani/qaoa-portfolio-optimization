"""Shared pytest setup: repo root on sys.path for ``src.*`` imports."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
