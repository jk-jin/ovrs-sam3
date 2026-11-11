from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PACKAGE_NAME = "_reca_test_pkg"


def load_module(relative_name: str):
    if PACKAGE_NAME not in sys.modules:
        pkg = types.ModuleType(PACKAGE_NAME)
        pkg.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = pkg
    return importlib.import_module(f"{PACKAGE_NAME}.{relative_name}")
