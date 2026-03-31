from __future__ import annotations

import copy
import runpy
from pathlib import Path
from typing import Any, Dict, Iterable


class Config(dict):
    """Tiny python-config loader with mmseg-like `_base_` support."""

    @classmethod
    def fromfile(cls, filename: str | Path) -> 'Config':
        filename = Path(filename).resolve()
        cfg_dict = cls._load_recursive(filename)
        return cls(cfg_dict)

    @classmethod
    def _load_recursive(cls, filename: Path) -> Dict[str, Any]:
        namespace = runpy.run_path(str(filename))
        base_files = namespace.pop('_base_', [])
        if isinstance(base_files, str):
            base_files = [base_files]

        merged: Dict[str, Any] = {}
        for base_file in base_files:
            base_path = (filename.parent / base_file).resolve()
            base_cfg = cls._load_recursive(base_path)
            merged = cls._merge_dicts(merged, base_cfg)

        current_cfg = {
            k: v for k, v in namespace.items()
            if not k.startswith('__') and k not in ('_base_',)
        }
        merged = cls._merge_dicts(merged, current_cfg)
        return merged

    @classmethod
    def _merge_dicts(cls, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = cls._merge_dicts(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out
