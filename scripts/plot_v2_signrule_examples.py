"""Plot May v2 Tanaka rows selected by the historical sign-transition rule."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notes/v2_old_signrule_examples_20260723.png"
RELATIVE_DEADZONE = 0.03

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class ExampleSpec:
    source: str
    archive: Path
    batch: int
    row: int
    expected_sign_count: int
    count_class: str


@dataclass(frozen=True)
class Example:
    spec: ExampleSpec
    case_id: int
    time: float
    depth: float
    x: FloatArray
    eta: FloatArray
    xi: FloatArray
    q: FloatArray
    forward_qx: FloatArray
    sign_count: int


EXAMPLE_SPECS = (
    ExampleSpec(
        source="adaptive g1",
        archive=ROOT / "data/tanaka_2_adaptive_g1.npz",
        batch=0,
        row=3107,
        expected_sign_count=12,
        count_class="marginal",
    ),
    ExampleSpec(
        source="adaptive g0",
        archive=ROOT / "data/tanaka_2_adaptive_g0.npz",
        batch=0,
        row=3329,
        expected_sign_count=12,
        count_class="marginal",
    ),
    ExampleSpec(
        source="adaptive g1",
        archive=ROOT / "data/tanaka_2_adaptive_g1.npz",
        batch=0,
        row=32578,
        expected_sign_count=48,
        count_class="high",
    ),
    ExampleSpec(
        source="adaptive g0",
        archive=ROOT / "data/tanaka_2_adaptive_g0.npz",
        batch=0,
        row=6276,
        expected_sign_count=162,
        count_class="maximum in audited g0 batch",
    ),
)


def read_array(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with archive.open(name) as handle:
        return np.load(handle, allow_pickle=False)


def count_sign_changes(
    q: FloatArray,
    relative_deadzone: float = RELATIVE_DEADZONE,
) -> int:
    difference = np.roll(q, -1) - q
    threshold = relative_deadzone * float(np.max(np.abs(difference)))
    signs = np.where(
        difference > threshold,
        1,
        np.where(difference < -threshold, -1, 0),
    )
    active_signs = signs[signs != 0]
    if active_signs.size <= 1:
        return 0
    return int(np.sum(active_signs != np.roll(active_signs, -1)))


def load_examples(specs: tuple[ExampleSpec, ...]) -> list[Example]:
    examples: list[Example] = []
    archives = {spec.archive for spec in specs}
    for archive_path in sorted(archives):
        archive_specs = [spec for spec in specs if spec.archive == archive_path]
        with zipfile.ZipFile(archive_path) as archive:
            x = np.asarray(read_array(archive, "x.npy"), dtype=np.float64)
            for batch in sorted({spec.batch for spec in archive_specs}):
                batch_specs = [spec for spec in archive_specs if spec.batch == batch]
                rows = np.asarray([spec.row for spec in batch_specs], dtype=np.int64)
                suffix = f"batch_{batch:04d}.npy"
                case_ids = read_array(archive, f"case_id_{suffix}")[rows]
                times = read_array(archive, f"time_{suffix}")[rows]
                depths = read_array(archive, f"depth_{suffix}")[rows]
                eta = read_array(archive, f"eta_{suffix}")[rows]
                xi = read_array(archive, f"xi_{suffix}")[rows]
                q = read_array(archive, f"gxi_{suffix}")[rows]
                dx = float(x[1] - x[0])

                for index, spec in enumerate(batch_specs):
                    q_row = np.asarray(q[index], dtype=np.float64)
                    sign_count = count_sign_changes(q_row)
                    if sign_count != spec.expected_sign_count:
                        raise ValueError(
                            f"{archive_path.name} row {spec.row}: expected "
                            f"C={spec.expected_sign_count}, observed C={sign_count}"
                        )
                    examples.append(
                        Example(
                            spec=spec,
                            case_id=int(case_ids[index]),
                            time=float(times[index]),
                            depth=float(depths[index]),
                            x=x,
                            eta=np.asarray(eta[index], dtype=np.float64),
                            xi=np.asarray(xi[index], dtype=np.float64),
                            q=q_row,
                            forward_qx=(np.roll(q_row, -1) - q_row) / dx,
                            sign_count=sign_count,
                        )
                    )

    order = {spec: index for index, spec in enumerate(specs)}
    return sorted(examples, key=lambda example: order[example.spec])


def format_scale(field: FloatArray) -> str:
    return rf"$\max|\cdot|={np.max(np.abs(field)):.2e}$"


def plot_examples(examples: list[Example], output: Path) -> None:
    figure, axes = plt.subplots(
        len(examples),
        4,
        figsize=(15.0, 11.5),
        sharex=True,
        constrained_layout=True,
    )
    column_titles = (
        r"surface elevation $\eta$",
        r"surface potential $\xi$",
        r"stored $q=G(\eta)\xi$",
        r"cyclic forward difference $D^+q$",
    )
    for axis, title in zip(axes[0], column_titles, strict=True):
        axis.set_title(title)

    for row, example in enumerate(examples):
        fields = (example.eta, example.xi, example.q, example.forward_qx)
        for column, field in enumerate(fields):
            axis = axes[row, column]
            axis.plot(example.x, field, color="tab:blue", linewidth=0.9)
            axis.grid(alpha=0.2)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            axis.text(
                0.98,
                0.94,
                format_scale(field),
                ha="right",
                va="top",
                transform=axis.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )

        deadzone = RELATIVE_DEADZONE * float(
            np.max(np.abs(example.forward_qx))
        )
        axes[row, 3].axhline(
            deadzone,
            color="tab:red",
            linewidth=0.8,
            linestyle="--",
        )
        axes[row, 3].axhline(
            -deadzone,
            color="tab:red",
            linewidth=0.8,
            linestyle="--",
        )
        axes[row, 0].set_ylabel(
            "\n".join(
                (
                    f"{example.spec.source}; {example.spec.count_class}",
                    (
                        f"batch {example.spec.batch}, row {example.spec.row}, "
                        f"case {example.case_id}"
                    ),
                    (
                        f"t={example.time:.2f}, h={example.depth:.5f}, "
                        f"C={example.sign_count}"
                    ),
                )
            ),
            fontsize=8,
        )

    for axis in axes[-1]:
        axis.set_xlabel(r"$x$")

    figure.suptitle(
        "May v2 Tanaka rows that would fail the historical sign rule",
        fontsize=15,
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.07, 1.0, 0.91))
    figure.text(
        0.5,
        0.01,
        (
            "Raw values: no field has been normalized.  The annotation in each "
            "panel gives its raw maximum magnitude.  "
            r"$q$ is the stored, hard-filtered label "
            r"($|k|\leq128$, filter_fraction=0.25)."
            "\n"
            r"The historical statistic $C=C_{0.03}(q)$ counts cyclic sign "
            r"transitions of $D^+q$ after discarding values between the red "
            "dashed lines."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    examples = load_examples(EXAMPLE_SPECS)
    plot_examples(examples, OUTPUT)
    for example in examples:
        print(
            f"{example.spec.archive.name}: "
            f"batch_{example.spec.batch:04d}[{example.spec.row}], "
            f"case={example.case_id}, t={example.time:.8g}, "
            f"h={example.depth:.8g}, C={example.sign_count}"
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
