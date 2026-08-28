## General Coding
- Keep diffs minimal and localized. Change only what is needed for the task.
- Do not mix unrelated cleanup into functional changes.
- Prefer the smallest correct change, not the smallest possible edit.
- A function does not need to return its inputs to the caller if they are not mutated during the function call.

## Python

### Code Conventions
- Prefer functional code (map, filter) over imperative code for simple stateless transformations. As a general rule it should mostly improve readability over the imperative, not drastically alter functioanlity.
- When parsing may fail and the fallback is a fixed default, initialize the default first, then overwrite it inside a narrow `with suppress(...)` block. Avoid combining presence checks, parsing, and fallback assignment in one `try/except`.
- Assign types to all function arguments and return types. If the return types become unwiedly it is ok to construct type aliases that are still semantically meaningful. If you are certain the full type is inferrable it is also ok to shorten the type signature.

### Environment + Tooling
- DO NOT globally install Python packages.
- Prefer to use uv over other alternatives for helping scaffold new python projects
- If uv is present prefer to use `uv run ...` instead of `python -m uvicorn`. E.g. `uv run my_python.py`
- If ruff is installed use `uv run ruff` to verify correctness frequently.
- If pyright is installed use `uv run pyright` to verify correctness frequently.
- If the tooling is not available, try to run python -m py_compile or whatever the equivalent is.
- Commit file changes liberally to git. Do not use other methods of ensuring reproducibility.

### Testing
- Prefer to use hypothesis when there are clear invariants that should hold.
- Avoid tests for trivial cases and minor behavior.

### Fetching from ArXiv
Try hitting the Ar5iv API to grab a nice readable HTML instead of trying to use some OCR tool to read the pdf directly: [https://ar5iv.labs.arxiv.org/](https://ar5iv.labs.arxiv.org/)

### Experiments
- Keep a log of all experiments in `EXPERIMENTS.md` with correctness, timing, and merge decision.
- Use local `HH:MM` time from `date` when adding new rows.
- Use `--:--` only for older rows where exact time was not recorded; do not invent precision.
- Fill both `Motivation` and `What Tried / Evidence`.
- `Motivation` should say why this was plausible before running it.
- `What Tried / Evidence` should say what changed and what the result taught us.
- Keep timings in `HH:MM` format.
