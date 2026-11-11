from __future__ import annotations

from typing import Callable, Type

DATASETS: dict[str, Type] = {}


def register_dataset(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        if name in DATASETS:
            raise KeyError(f"dataset '{name}' already registered")
        cls.name = name
        DATASETS[name] = cls
        return cls

    return deco
