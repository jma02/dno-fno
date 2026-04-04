__all__ = [
    "CrestSpec",
    "build_multi_crest_initial_condition",
    "save_multi_crest_initial_condition",
]


def __getattr__(name: str):
    if name in __all__:
        from .multi_crest import (
            CrestSpec,
            build_multi_crest_initial_condition,
            save_multi_crest_initial_condition,
        )

        exports = {
            "CrestSpec": CrestSpec,
            "build_multi_crest_initial_condition": build_multi_crest_initial_condition,
            "save_multi_crest_initial_condition": save_multi_crest_initial_condition,
        }
        return exports[name]
    raise AttributeError(name)
