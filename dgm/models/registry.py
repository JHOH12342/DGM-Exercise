from __future__ import annotations

from collections.abc import Callable

MODELS: dict[str, type] = {}


def register_model(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in MODELS:
            raise KeyError(f"model '{name}' already registered")
        cls.name = name
        MODELS[name] = cls
        return cls

    return deco
