from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# NumPy RNG — safe serialization (no ndarray/dtype objects in checkpoint)
# ---------------------------------------------------------------------------

def _capture_numpy_rng_state() -> Dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()

    return {
        "algorithm": str(algorithm),
        "keys": torch.from_numpy(
            keys.astype(np.int64, copy=True)
        ),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng_state(state: Dict[str, Any]) -> None:
    required_keys = {
        "algorithm",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    missing = required_keys - state.keys()
    if missing:
        raise KeyError(f"NumPy RNG state missing fields: {sorted(missing)}")

    keys_tensor = state["keys"]
    if not isinstance(keys_tensor, torch.Tensor):
        raise TypeError("NumPy RNG 'keys' must be a torch.Tensor.")
    if keys_tensor.ndim != 1:
        raise ValueError(
            f"NumPy RNG 'keys' must be 1-D, got {tuple(keys_tensor.shape)}."
        )

    keys = (
        keys_tensor.detach()
        .cpu()
        .numpy()
        .astype(np.uint32, copy=True)
    )

    np.random.set_state((
        str(state["algorithm"]),
        keys,
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _capture_numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda"] = None
    else:
        state["torch_cuda"] = None

    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        _restore_numpy_rng_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
