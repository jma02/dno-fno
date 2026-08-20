"""Check the inert literature-aligned paper-corpus release-file inventory."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import stat
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[1]
MANIFEST: Final = ROOT / "reproducibility/paper_corpus_release_files.json"
ALLOWED_KINDS: Final = frozenset(
    {"python", "shell", "shell_library", "json", "text", "checksums"}
)
SUFFIX_BY_KIND: Final = {
    "python": ".py",
    "shell": ".sh",
    "shell_library": ".sh",
    "json": ".json",
    "text": ".md",
}
REQUIRED_PUBLICATION_IMPORTS: Final = {
    "scripts/run_paper_corpus_quota.py": (
        "solver.gen_data.stokes_quota_executor",
        "solver/gen_data/stokes_quota_executor.py",
    ),
    "scripts/render_paper_corpus_worst_cases.py": (
        "solver.gen_data.clean_tanaka_dataset",
        "solver/gen_data/clean_tanaka_dataset.py",
    ),
    "train-jax-10m/util.py": (
        "jax_training_util",
        "jax_training_util.py",
    ),
}
REQUIRED_EAGER_SOLVER_PACKAGE_PATHS: Final = frozenset(
    {
        "solver/__init__.py",
        "solver/data/__init__.py",
        "solver/data/solitary_loader_jax.py",
        "solver/evals/__init__.py",
        "solver/evals/compare_rollout_jax.py",
        "solver/evals/render_rollout_movie.py",
        "solver/gen_data/__init__.py",
        "solver/solvers/__init__.py",
        "solver/tanaka_ICs/__init__.py",
    }
)
REQUIRED_EAGER_TRAINER_IMPORTS: Final = {
    "dno_net": "models/dno-net/dno_net.py",
    "dno_net_v2": "models/dno-net/dno_net_v2.py",
    "fno1d": "models/fno-jax/fno1d.py",
    "losses": "models/fno-jax/losses.py",
    "eval_on_dno_dataset": "train-jax-10m/eval_on_dno_dataset.py",
    "finite_time_phase_regularizer": ("train-jax-10m/finite_time_phase_regularizer.py"),
    "hadamard_shape_regularizer": "train-jax-10m/hadamard_shape_regularizer.py",
    "modal_phase_rate_regularizer": ("train-jax-10m/modal_phase_rate_regularizer.py"),
    "mode_balanced_regularizer": "train-jax-10m/mode_balanced_regularizer.py",
    "source_conditioning": "train-jax-10m/source_conditioning.py",
    "stage_tangent_regularizer": "train-jax-10m/stage_tangent_regularizer.py",
    "translation_tangent_regularizer": (
        "train-jax-10m/translation_tangent_regularizer.py"
    ),
}


def _load_manifest() -> dict[str, object]:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise TypeError("release-file manifest must contain a JSON object")
    return record


def _load_entries() -> tuple[dict[str, object], ...]:
    record = _load_manifest()
    entries = record.get("files")
    if not isinstance(entries, list):
        raise TypeError("release-file manifest must contain a files list")
    if not all(isinstance(entry, dict) for entry in entries):
        raise TypeError("every release-file entry must be a JSON object")
    return tuple(entries)


def _imported_modules(relative: str) -> set[str]:
    tree = ast.parse(
        (ROOT / relative).read_text(encoding="utf-8"),
        filename=relative,
    )
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
    return imported_modules


def _module_level_imported_modules(relative: str) -> set[str]:
    tree = ast.parse(
        (ROOT / relative).read_text(encoding="utf-8"),
        filename=relative,
    )
    return {
        name
        for node in tree.body
        for name in (
            tuple(alias.name for alias in node.names)
            if isinstance(node, ast.Import)
            else (node.module,)
            if isinstance(node, ast.ImportFrom) and node.module is not None
            else ()
        )
    }


class PaperCorpusReleaseFileManifestTest(unittest.TestCase):
    def test_publication_closure_is_exact_and_reproducible(self) -> None:
        record = _load_manifest()
        closure = record.get("publication_closure")
        self.assertIsInstance(closure, dict)
        assert isinstance(closure, dict)

        count = closure.get("count")
        serialization = closure.get("path_serialization")
        expected_sha256 = closure.get("sorted_path_set_sha256")
        paths = closure.get("paths")
        self.assertEqual(
            closure.get("purpose"),
            "Exact coherent code-repository publication closure: release "
            "integration files, current JONSWAP/Tanaka/Benjamin-Feir numerical "
            "source and dependency paths, recovered historical Stokes/Benjamin-"
            "Feir source bytes, and supplemental import, document, and test paths. "
            "It intentionally excludes the ignored private paper/ tree; its source "
            "refresh, figure installation, and compile verification are separately "
            "governed by "
            "notes/paper_corpus_generation_readiness_20260726.md.",
        )
        self.assertIs(type(count), int)
        self.assertEqual(count, 141)
        self.assertEqual(
            serialization,
            "Sort paths by Unicode code point, encode each path as UTF-8, and "
            "terminate every path, including the last, with LF.",
        )
        self.assertEqual(
            expected_sha256,
            "2de26512aee1fc4819cc6dd6364ac02a7eda958d3c93c54903e14a730f764bb8",
        )
        self.assertIsInstance(paths, list)
        assert isinstance(count, int)
        assert isinstance(expected_sha256, str)
        assert isinstance(paths, list)
        self.assertTrue(all(isinstance(path, str) for path in paths))
        assert all(isinstance(path, str) for path in paths)
        self.assertEqual(len(paths), count)
        self.assertEqual(paths, sorted(paths))
        self.assertEqual(len(paths), len(set(paths)))

        for relative in paths:
            path = Path(relative)
            self.assertFalse(path.is_absolute(), relative)
            self.assertNotIn("..", path.parts, relative)
            self.assertTrue((ROOT / path).is_file(), relative)

        serialized = "".join(f"{path}\n" for path in paths).encode("utf-8")
        self.assertEqual(hashlib.sha256(serialized).hexdigest(), expected_sha256)
        release_paths = {str(entry["path"]) for entry in _load_entries()}
        self.assertLessEqual(release_paths, set(paths))

    def test_publication_closure_contains_required_local_imports(self) -> None:
        closure = _load_manifest()["publication_closure"]
        assert isinstance(closure, dict)
        paths = closure["paths"]
        assert isinstance(paths, list)
        closure_paths = set(paths)

        for importer, (module, dependency) in REQUIRED_PUBLICATION_IMPORTS.items():
            with self.subTest(importer=importer, dependency=dependency):
                self.assertIn(importer, closure_paths)
                self.assertIn(dependency, closure_paths)
                self.assertIn(module, _imported_modules(importer))

    def test_publication_closure_preserves_eager_import_semantics(self) -> None:
        closure = _load_manifest()["publication_closure"]
        assert isinstance(closure, dict)
        paths = closure["paths"]
        assert isinstance(paths, list)
        closure_paths = set(paths)

        self.assertLessEqual(REQUIRED_EAGER_SOLVER_PACKAGE_PATHS, closure_paths)
        trainer_imports = _module_level_imported_modules(
            "train-jax-10m/1d_dno_fno_jax.py"
        )
        for module, dependency in REQUIRED_EAGER_TRAINER_IMPORTS.items():
            with self.subTest(module=module, dependency=dependency):
                self.assertIn(module, trainer_imports)
                self.assertIn(dependency, closure_paths)

    def test_manifest_paths_exist_and_have_declared_types(self) -> None:
        entries = _load_entries()
        paths: list[str] = []

        for entry in entries:
            relative = entry.get("path")
            kind = entry.get("kind")
            role = entry.get("role")
            self.assertIsInstance(relative, str)
            self.assertIsInstance(kind, str)
            self.assertIsInstance(role, str)
            assert isinstance(relative, str)
            assert isinstance(kind, str)
            assert isinstance(role, str)

            path = Path(relative)
            self.assertFalse(path.is_absolute(), relative)
            self.assertNotIn("..", path.parts, relative)
            self.assertIn(kind, ALLOWED_KINDS, relative)
            self.assertTrue(role.strip(), relative)
            self.assertTrue((ROOT / path).is_file(), relative)
            paths.append(relative)

            if kind in SUFFIX_BY_KIND:
                self.assertEqual(path.suffix, SUFFIX_BY_KIND[kind], relative)
            elif kind == "checksums":
                self.assertEqual(path.name, "SHA256SUMS", relative)

        self.assertEqual(len(paths), len(set(paths)), "manifest paths must be unique")

    def test_python_and_json_files_are_syntactically_valid(self) -> None:
        for entry in _load_entries():
            relative = str(entry["path"])
            path = ROOT / relative
            if entry["kind"] == "python":
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            elif entry["kind"] == "json":
                json.loads(path.read_text(encoding="utf-8"))

    def test_shell_entrypoints_are_bash_and_executable(self) -> None:
        shell_entries = tuple(
            entry for entry in _load_entries() if entry["kind"] == "shell"
        )
        self.assertTrue(shell_entries, "manifest must name release shell entrypoints")
        for entry in shell_entries:
            relative = str(entry["path"])
            path = ROOT / relative
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(first_line, "#!/usr/bin/env bash", relative)
            self.assertIs(entry.get("executable"), True, relative)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755, relative)

    def test_shell_libraries_are_bash_and_source_only(self) -> None:
        libraries = tuple(
            entry for entry in _load_entries() if entry["kind"] == "shell_library"
        )
        self.assertTrue(libraries, "manifest must name sourced shell libraries")
        for entry in libraries:
            relative = str(entry["path"])
            path = ROOT / relative
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(first_line, "#!/usr/bin/env bash", relative)
            self.assertIs(entry.get("executable"), False, relative)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644, relative)


if __name__ == "__main__":
    unittest.main()
