from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modified_tanaka import (
        ModifiedTanakaBatchSolution,
        ModifiedTanakaParams,
        ModifiedTanakaSeed,
        ModifiedTanakaSolution,
        make_default_tanaka_template,
        make_tanaka_seed,
        solve_modified_tanaka,
        solve_modified_tanaka_batched,
        solve_tanaka_branch,
    )

__all__ = [
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


def __getattr__(name: str) -> object:
    if name in __all__:
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

        exports: dict[str, object] = {
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
