"""Load tensor/data dependencies only when a data object is requested."""
from importlib import import_module

__all__ = [
    'build_dataloader',
    'build_dataset',
    'OVSemanticCollator',
    'OVSemanticSegDataset',
]


def __getattr__(name):
    modules = dict(build_dataloader="build", build_dataset="build",
                   OVSemanticCollator="collate", OVSemanticSegDataset="dataset")
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f".{modules[name]}", __name__), name)
    globals()[name] = value
    return value
