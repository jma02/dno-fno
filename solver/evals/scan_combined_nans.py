"""Single-process NaN/inf scan of combined_dataset.npz, per source.

Outputs:
    /tmp/combined_nan_report.txt      per-source NaN/inf row counts (eta, xi, gxi)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SRC = Path("data/combined_dataset.npz")
META = Path("data/combined_dataset.meta.json")
OUT = Path("/tmp/combined_nan_report.txt")
CHUNK = 50_000


def main() -> None:
    meta = json.loads(META.read_text())
    legend: dict[int, str] = {int(k): v for k, v in meta["source_legend"].items()}
    n_total = int(meta["n_samples"])
    print(f"combined: N={n_total:,}, chunk={CHUNK:,}")

    archive = np.load(SRC, mmap_mode="r")
    eta_all = archive["eta"]; xi_all = archive["xi"]; gxi_all = archive["gxi"]
    src_all = archive["source"]

    # per-source counters: nan_eta, nan_xi, nan_gxi, inf_eta, inf_xi, inf_gxi, n
    counters = {sid: [0]*7 for sid in legend}

    n_chunks = (n_total + CHUNK - 1) // CHUNK
    for ci, s in enumerate(range(0, n_total, CHUNK)):
        e = min(s + CHUNK, n_total)
        eta = np.asarray(eta_all[s:e], dtype=np.float32)
        xi  = np.asarray(xi_all[s:e],  dtype=np.float32)
        gxi = np.asarray(gxi_all[s:e], dtype=np.float32)
        src = np.asarray(src_all[s:e], dtype=np.int8)

        nan_e_row = np.isnan(eta).any(axis=1)
        nan_x_row = np.isnan(xi).any(axis=1)
        nan_g_row = np.isnan(gxi).any(axis=1)
        inf_e_row = np.isinf(eta).any(axis=1)
        inf_x_row = np.isinf(xi).any(axis=1)
        inf_g_row = np.isinf(gxi).any(axis=1)

        for sid in np.unique(src):
            m = src == sid
            sid_i = int(sid)
            counters[sid_i][0] += int(nan_e_row[m].sum())
            counters[sid_i][1] += int(nan_x_row[m].sum())
            counters[sid_i][2] += int(nan_g_row[m].sum())
            counters[sid_i][3] += int(inf_e_row[m].sum())
            counters[sid_i][4] += int(inf_x_row[m].sum())
            counters[sid_i][5] += int(inf_g_row[m].sum())
            counters[sid_i][6] += int(m.sum())

        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            print(f"  chunk {ci+1}/{n_chunks} ({e:,}/{n_total:,})")

    lines = [f"{'source':<22} | {'n':>10} | {'nan_eta':>8} {'nan_xi':>8} {'nan_gxi':>8} | "
             f"{'inf_eta':>8} {'inf_xi':>8} {'inf_gxi':>8}",
             "-" * 100]
    grand = [0]*7
    for sid in sorted(legend.keys()):
        c = counters[sid]
        for j in range(7):
            grand[j] += c[j]
        lines.append(f"{legend[sid]:<22} | {c[6]:>10,} | "
                     f"{c[0]:>8,} {c[1]:>8,} {c[2]:>8,} | "
                     f"{c[3]:>8,} {c[4]:>8,} {c[5]:>8,}")
    lines.append("-" * 100)
    lines.append(f"{'TOTAL':<22} | {grand[6]:>10,} | "
                 f"{grand[0]:>8,} {grand[1]:>8,} {grand[2]:>8,} | "
                 f"{grand[3]:>8,} {grand[4]:>8,} {grand[5]:>8,}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"  -> {OUT}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
