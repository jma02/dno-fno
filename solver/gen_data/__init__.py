__all__ = [
    "CrestSpec",
    "available_reference_amplitudes",
    "build_multi_crest_initial_condition",
    "build_multi_crest_preset",
    "make_preset_specs",
    "save_multi_crest_initial_condition",
]


def __getattr__(name: str):
    if name in __all__:
        from .multi_crest import (
            CrestSpec,
            available_reference_amplitudes,
            build_multi_crest_initial_condition,
            build_multi_crest_preset,
            make_preset_specs,
            save_multi_crest_initial_condition,
        )

        exports = {
            "CrestSpec": CrestSpec,
            "available_reference_amplitudes": available_reference_amplitudes,
            "build_multi_crest_initial_condition": build_multi_crest_initial_condition,
            "build_multi_crest_preset": build_multi_crest_preset,
            "make_preset_specs": make_preset_specs,
            "save_multi_crest_initial_condition": save_multi_crest_initial_condition,
        }
        return exports[name]
    raise AttributeError(name)
