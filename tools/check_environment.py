#!/usr/bin/env python3
"""Lightweight environment check for the public GDIS-Bio repository."""

from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = [
    "numpy",
    "pandas",
    "scipy",
    "scanpy",
    "anndata",
    "matplotlib",
    "sklearn",
    "openpyxl",
    "requests",
    "pygdis",
]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    failed = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "available")
            print(f"OK  {name}: {version}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    if failed:
        print("\nEnvironment check failed.")
        return 1
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
