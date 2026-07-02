"""Build a minimal gallery site: one figure per regime, plus an optional
Tanaka-IC sensitivity section.

For each regime, displays a single PNG with two stacked panels:
  - top: relative L2 η-error vs time, for three ICs binned by initial
         mean wavenumber (low / mid / high)
  - bottom: mean wavenumber of the truth ⟨k⟩(t) for the same three ICs

PNGs are produced by `plot_error_vs_k_time.py`. If `--tanaka_sens_dir` is
supplied and exists, an additional section shows the FNO Tanaka case-0
sensitivity sweep: the 3-panel summary plot plus a grid of GIFs for each
(amplitude / substeps / Nx) config.

Usage:
    uv run python -m playground.plot_error_vs_k_time \\
        --run_dir outputs/fno_w128b6_v3_hclip5_20260514_063811 \\
        --output_dir outputs/error_vs_k

    uv run python -m playground.build_paper_dashboard \\
        --run_name "FNO baseline" \\
        --kt_dir outputs/error_vs_k \\
        --tanaka_sens_dir playground/tanaka_sensitivity \\
        --output_dir playground/paper_dashboard
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REGIMES = [
    "tanaka_g0", "tanaka_g1",
    "bf_g0", "bf_g1", "bf_modal",
    "linear",
    "stokes_deep", "stokes_finite",
    "random_sea_deep", "random_sea_finite",
]


SENS_AXIS_LABELS: dict[str, tuple[str, str]] = {
    "amp": ("Wave amplitude (physical)",
            "IC scaled by α; truth regenerated from the scaled IC."),
    "sub": ("Integrator substeps (numerical dt)",
            "More substeps → finer integration step. NaN means GL2 itself is unstable at that dt."),
    "nx":  ("Spatial resolution Nx (numerical)",
            "IC bandlimited-resampled; FNO applied at the new grid."),
}


def _gif_label(stem: str) -> str:
    if stem.startswith("amp_alpha"):
        return f"α = {stem[len('amp_alpha'):]}"
    if stem.startswith("sub_substeps"):
        return f"substeps = {int(stem[len('sub_substeps'):])}"
    if stem.startswith("nx_Nx"):
        return f"Nx = {stem[len('nx_Nx'):]}"
    return stem


def sensitivity_section(gif_files: list[Path], have_summary: bool) -> str:
    if not gif_files and not have_summary:
        return ""
    by_axis: dict[str, list[Path]] = {"amp": [], "sub": [], "nx": []}
    for p in sorted(gif_files):
        key = p.stem.split("_", 1)[0]
        if key in by_axis:
            by_axis[key].append(p)

    parts: list[str] = []
    parts.append('<section class="sens">')
    parts.append('  <h2>Tanaka case 0 — sensitivity to physical and numerical parameters</h2>')
    parts.append(
        '  <p class="lede">'
        'Each panel below sweeps one knob (IC amplitude, integrator substeps, spatial resolution) '
        'with the others held at the training-grid baseline (α=1, substeps=8, Nx=1024). '
        'Truth-vs-FNO trajectories on the left as GIFs; the 3-panel summary plots the rel-L² η-error '
        'over time for every config.'
        '</p>')
    if have_summary:
        parts.append('  <figure class="summary">')
        parts.append('    <img src="sens/fno_tanaka_sensitivity.png" alt="FNO Tanaka sensitivity summary" loading="lazy">')
        parts.append('    <figcaption>rel-L² η-error vs t across the three sweep axes. Each curve is one config.</figcaption>')
        parts.append('  </figure>')

    for axis, (heading, sub) in SENS_AXIS_LABELS.items():
        gifs = by_axis.get(axis, [])
        if not gifs:
            continue
        parts.append('  <div class="axis">')
        parts.append(f'    <h3>{heading}</h3>')
        parts.append(f'    <p class="axis-sub">{sub}</p>')
        parts.append('    <div class="gif-grid">')
        for p in gifs:
            label = _gif_label(p.stem)
            parts.append(
                f'      <figure class="gif-card">'
                f'<img src="sens/gifs/{p.name}" alt="{p.stem}" loading="lazy">'
                f'<figcaption>{label}</figcaption>'
                f'</figure>')
        parts.append('    </div>')
        parts.append('  </div>')
    parts.append('</section>')
    return "\n".join(parts)


def index_html(run_name: str, regimes_with_ics: list[tuple[str, bool]], sens_block: str) -> str:
    blocks = []
    for r, has_ic in regimes_with_ics:
        ic_img = (
            f'\n    <img class="ic" src="kt/{r}_ics.png" alt="{r} initial conditions" loading="lazy">\n'
            f'    <figcaption class="ic-cap">'
            f'IC visualisation: η(x,0), ξ(x,0), |η̂(k,0)|² for the three ICs picked by initial centroid wavenumber.'
            f'</figcaption>'
        ) if has_ic else ""
        blocks.append(
            f'  <figure>{ic_img}\n'
            f'    <img src="kt/{r}_err_vs_k.png" alt="{r} error" loading="lazy">\n'
            f'  </figure>'
        )
    figures = "\n".join(blocks)
    return f"""<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<title>{run_name} — rollout error vs initial wavenumber</title>
<style>
  :root {{ --fg: #111; --muted: #555; --bg: #fafafa; --line: #e3e3e3; }}
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; color: var(--fg); background: var(--bg); margin: 0; padding: 28px 32px 64px; }}
  header, main {{ max-width: 1100px; margin: 0 auto; }}
  header {{ margin-bottom: 20px; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; font-weight: 600; }}
  h2 {{ font-size: 18px; margin: 32px 0 8px; font-weight: 600; }}
  h3 {{ font-size: 15px; margin: 18px 0 4px; font-weight: 600; color: #333; }}
  p.lede {{ color: var(--muted); font-size: 14px; line-height: 1.5; margin: 0 0 8px; max-width: 820px; }}
  p.axis-sub {{ color: var(--muted); font-size: 13px; margin: 0 0 10px; }}
  main {{ display: grid; gap: 28px; }}
  figure {{ margin: 0; background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 10px; }}
  figure img {{ width: 100%; height: auto; display: block; }}
  figure img.ic {{ margin-bottom: 4px; border-bottom: 1px dashed var(--line); padding-bottom: 6px; }}
  figcaption {{ font-size: 12px; color: var(--muted); margin-top: 4px; text-align: center; }}
  figcaption.ic-cap {{ margin: 2px 0 10px; font-style: italic; }}
  section.sens {{ background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 18px 22px 24px; }}
  section.sens .summary img {{ max-height: 360px; object-fit: contain; }}
  section.sens .axis {{ margin-top: 18px; }}
  .gif-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }}
  .gif-card {{ padding: 6px; }}
  .gif-card img {{ border-radius: 3px; }}
  hr {{ border: none; border-top: 1px solid var(--line); margin: 24px 0; }}
</style>
</head>
<body>
<header>
  <h1>{run_name} — rollout error vs initial wavenumber</h1>
  <p class=lede>
    For each regime, three ICs are picked by the energy-weighted spectral centroid of η at t=0
    (low = red, mid = blue, high = green). The upper image in each figure shows those ICs:
    η(x,0), ξ(x,0), and the |η̂(k,0)|² spectrum with each centroid marked. The lower image
    shows the relative L2 η-error of the surrogate's rollout (top, log-y) and the truth mean
    wavenumber ⟨k⟩(t) (bottom) over the same horizon.
  </p>
</header>
<main>
{sens_block}
{figures}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", default="FNO baseline")
    parser.add_argument("--kt_dir", default="outputs/error_vs_k")
    parser.add_argument("--tanaka_sens_dir", default="playground/tanaka_sensitivity",
                        help="If this directory contains fno_tanaka_sensitivity.png and/or gifs/*.gif, "
                             "they are copied into the dashboard and rendered as a Tanaka sensitivity "
                             "section above the per-regime grid.")
    parser.add_argument("--output_dir", default="playground/paper_dashboard")
    parser.add_argument("--regimes", default=",".join(REGIMES))
    args = parser.parse_args()

    kt_dir = Path(args.kt_dir).resolve()
    sens_dir = Path(args.tanaka_sens_dir).resolve() if args.tanaka_sens_dir else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous artifacts.
    for sub in ("heatmaps", "spectrum", "data", "kt", "sens"):
        d = output_dir / sub
        if d.exists():
            shutil.rmtree(d)
    dst = output_dir / "kt"
    dst.mkdir()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    available: list[tuple[str, bool]] = []
    for r in regimes:
        src = kt_dir / f"{r}_err_vs_k.png"
        if not src.exists():
            print(f"  skip {r}: {src.name} missing")
            continue
        shutil.copy2(src, dst / src.name)
        ic_src = kt_dir / f"{r}_ics.png"
        has_ic = ic_src.exists()
        if has_ic:
            shutil.copy2(ic_src, dst / ic_src.name)
        available.append((r, has_ic))

    sens_block = ""
    if sens_dir is not None and sens_dir.exists():
        sens_dst = output_dir / "sens"
        sens_dst.mkdir()
        have_summary = False
        summary = sens_dir / "fno_tanaka_sensitivity.png"
        if summary.exists():
            shutil.copy2(summary, sens_dst / summary.name)
            have_summary = True
        gif_files: list[Path] = []
        gif_src = sens_dir / "gifs"
        if gif_src.exists():
            (sens_dst / "gifs").mkdir(exist_ok=True)
            for g in sorted(gif_src.glob("*.gif")):
                shutil.copy2(g, sens_dst / "gifs" / g.name)
                gif_files.append(sens_dst / "gifs" / g.name)
        sens_block = sensitivity_section(gif_files, have_summary)
        print(f"  tanaka sensitivity: summary={have_summary}, gifs={len(gif_files)}")

    (output_dir / "index.html").write_text(
        index_html(args.run_name, available, sens_block), encoding="utf-8"
    )
    print(f"wrote {output_dir / 'index.html'} with {len(available)}/{len(regimes)} regimes")


if __name__ == "__main__":
    main()
