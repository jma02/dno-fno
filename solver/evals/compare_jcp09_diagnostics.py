"""Side-by-side JCP09 diagnostic comparison across multiple trajs.npz inputs.

Used to compare v8 baseline vs experiment A (fp64 model inference) vs
experiment B (filter_gxi eval) — per the cs_dno_jcp09_improvement_plan.md §4
decision tree.

Usage:
    uv run python -m solver.evals.compare_jcp09_diagnostics \
        --label baseline --trajs outputs/<run>/eval_suite_f64h/tanaka_g0_trajs.npz \
        --label A_f64m  --trajs outputs/<run>/eval_tanaka_f64m/tanaka_g0_trajs.npz \
        --label B_filter --trajs outputs/<run>/eval_tanaka_gxifilt/tanaka_g0_trajs.npz \
        [--t_diag 2.4] [--k_high_cut 30]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from solver.evals.compute_jcp09_diagnostics import compute_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", action="append", required=True,
                        help="Column label; pair-flag with --trajs.")
    parser.add_argument("--trajs", action="append", required=True,
                        help="trajs.npz path; pair-flag with --label.")
    parser.add_argument("--t_diag", type=float, default=2.4)
    parser.add_argument("--k_high_cut", type=int, default=30)
    args = parser.parse_args()

    if len(args.label) != len(args.trajs):
        raise SystemExit("--label and --trajs counts must match (use them in pairs)")

    diags = []
    for label, tp in zip(args.label, args.trajs):
        d = compute_diagnostics(Path(tp).resolve(),
                                t_diag=args.t_diag, k_high_cut=args.k_high_cut)
        diags.append((label, d))

    cids = [p["case_id"] for p in diags[0][1]["per_ic"]]
    depths = {p["case_id"]: p["h"] for p in diags[0][1]["per_ic"]}
    by_label = {label: {p["case_id"]: p for p in d["per_ic"]} for label, d in diags}

    print(f"\nt_diag={diags[0][1]['t_diag_actual']:.2f}  k_cut={args.k_high_cut}  PASS = err_high/truth_high < 0.1\n")
    header = f"{'cid':>10}  {'h':>6}  " + "  ".join(f"{lab:>12}" for lab, _ in diags)
    print(header)
    print("-" * len(header))

    fails_per = {lab: 0 for lab, _ in diags}
    nans_per = {lab: 0 for lab, _ in diags}
    for cid in cids:
        cells = []
        for lab, _ in diags:
            p = by_label[lab].get(cid)
            if p is None:
                cells.append(f"{'(missing)':>12}")
                continue
            if p["is_nan_final"]:
                nans_per[lab] += 1
                cells.append(f"{'NaN':>12}")
                continue
            r = p["err_high_over_truth_high_at_t_diag"]
            if r is None:
                cells.append(f"{'-':>12}")
                continue
            if r >= 0.1:
                fails_per[lab] += 1
                cells.append(f"{r:>11.3g}*")
            else:
                cells.append(f"{r:>12.3g}")
        print(f"{cid:>10}  {depths[cid]:>6.3f}  " + "  ".join(cells))

    print("-" * len(header))
    NB = len(cids)
    print(f"{'NaN':>10}  {'':>6}  " + "  ".join(
        f"{nans_per[lab]:>5}/{NB:<6}" for lab, _ in diags))
    print(f"{'FAIL>0.1':>10}  {'':>6}  " + "  ".join(
        f"{fails_per[lab]:>5}/{NB:<6}" for lab, _ in diags))


if __name__ == "__main__":
    main()
