from __future__ import annotations

import sys
import types
from importlib import import_module
from pathlib import Path

if __package__ in (None, ""):
    name = "_reca_suite"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[name] = package
    main = import_module(f"{name}.utils.experiment_suite").main
else:
    from ..utils.experiment_suite import main

if __name__ == "__main__":
    raise SystemExit(main("train"))
