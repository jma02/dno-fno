__all__ = [
    "build_amplitude_dataset",
    "make_default_tanaka_template",
    "ModifiedTanakaBatchSolution",
    "ModifiedTanakaParams",
    "ModifiedTanakaSolution",
    "ModifiedTanakaSeed",
    "make_tanaka_seed",
    "solve_modified_tanaka_batched",
    "solve_modified_tanaka",
    "solve_tanaka_branch",
]


def __getattr__(name: str):
    if name in __all__:
        from .build_amplitude_dataset import main as build_amplitude_dataset
        from .modified_tanaka import (
            make_default_tanaka_template,
            ModifiedTanakaBatchSolution,
            ModifiedTanakaParams,
            ModifiedTanakaSeed,
            ModifiedTanakaSolution,
            make_tanaka_seed,
            solve_modified_tanaka_batched,
            solve_modified_tanaka,
            solve_tanaka_branch,
        )

        exports = {
            "build_amplitude_dataset": build_amplitude_dataset,
            "make_default_tanaka_template": make_default_tanaka_template,
            "ModifiedTanakaBatchSolution": ModifiedTanakaBatchSolution,
            "ModifiedTanakaParams": ModifiedTanakaParams,
            "ModifiedTanakaSeed": ModifiedTanakaSeed,
            "ModifiedTanakaSolution": ModifiedTanakaSolution,
            "make_tanaka_seed": make_tanaka_seed,
            "solve_modified_tanaka_batched": solve_modified_tanaka_batched,
            "solve_modified_tanaka": solve_modified_tanaka,
            "solve_tanaka_branch": solve_tanaka_branch,
        }
        return exports[name]
    raise AttributeError(name)
