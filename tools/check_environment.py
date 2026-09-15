#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

from importlib import import_module
from importlib.metadata import version, PackageNotFoundError
import platform
import sys

REFERENCE = {
    "numpy": "2.4.6",
    "pandas": "2.3.3",
    "scipy": "1.17.1",
    "scanpy": "1.11.5",
    "anndata": "0.12.19",
    "pygdis": "1.0.0",
}
REQUIRED_IMPORTS = ["matplotlib", "sklearn", "requests"]

print(f"Python: {platform.python_version()}")
if sys.version_info[:2] != (3, 11):
    print("WARNING: reference analyses used Python 3.11.16")

failed = False
for pkg, expected in REFERENCE.items():
    try:
        observed = version(pkg)
    except PackageNotFoundError:
        print(f"MISSING: {pkg}")
        failed = True
        continue
    status = "OK" if observed == expected else "REFERENCE-DIFF"
    print(f"{status}: {pkg} {observed} (reference {expected})")
    if pkg == "pygdis" and observed != expected:
        failed = True

for module in REQUIRED_IMPORTS:
    try:
        import_module(module)
        print(f"OK: import {module}")
    except Exception as exc:
        print(f"MISSING/ERROR: import {module}: {exc}")
        failed = True

if failed:
    raise SystemExit(1)
print("Environment check completed.")
