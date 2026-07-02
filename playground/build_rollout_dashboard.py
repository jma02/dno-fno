"""Build a tiny static dashboard for interactive rollout + error-spectrum review.

Reads eval_suite/<regime>_trajs.npz from a run and emits:
  playground/rollout_dashboard/index.html
  playground/rollout_dashboard/data/<regime>.json

The HTML uses Plotly.js from CDN. Open the index in a browser to inspect
(η(x), |err(k)|²) at any (regime, IC, time) — no server needed beyond a static
file host. To deploy: `netlify deploy --dir=playground/rollout_dashboard`.

Data size is controlled by --x_stride and --t_stride; the defaults downsample
to ~1.5 MB per regime so all 10 fit under typical static-host limits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def downsample(arr: np.ndarray, t_stride: int, x_stride: int) -> np.ndarray:
    return arr[::t_stride, :, ::x_stride]


def build_regime_payload(
    npz_path: Path,
    length: float,
    t_stride: int,
    x_stride: int,
    max_ic: int,
) -> dict[str, object]:
    with np.load(npz_path) as d:
        times = np.asarray(d["times"]).astype(np.float32)
        truth_eta = np.asarray(d["truth_eta"]).astype(np.float32)
        pred_eta = np.asarray(d["pred_eta"]).astype(np.float32)
        case_ids = np.asarray(d["case_ids"]).astype(int) if "case_ids" in d.files else np.arange(truth_eta.shape[1])

    # Drop NaN-rollout ICs entirely so the dashboard never plots NaN curves.
    bad = ~np.isfinite(pred_eta).all(axis=(0, 2))
    if bad.any():
        keep = ~bad
        truth_eta = truth_eta[:, keep, :]
        pred_eta = pred_eta[:, keep, :]
        case_ids = case_ids[keep]
    n_t, n_b, nx = truth_eta.shape
    if max_ic and max_ic < n_b:
        truth_eta = truth_eta[:, :max_ic, :]
        pred_eta = pred_eta[:, :max_ic, :]
        case_ids = case_ids[:max_ic]
        n_b = max_ic

    # Spectral error per IC over the (full) k grid, then downsample t after.
    err = pred_eta - truth_eta  # (T, B, nx)
    err_power = np.abs(np.fft.rfft(err, axis=-1)) ** 2  # (T, B, K)
    truth_power = np.abs(np.fft.rfft(truth_eta, axis=-1)) ** 2
    n_k = err_power.shape[-1]
    k_phys = ((2.0 * np.pi / length) * np.arange(n_k)).astype(np.float32)

    # Downsample for transport.
    truth_eta_ds = downsample(truth_eta, t_stride, x_stride)
    pred_eta_ds = downsample(pred_eta, t_stride, x_stride)
    err_power_ds = err_power[::t_stride]  # keep all k
    truth_power_ds = truth_power[::t_stride]
    times_ds = times[::t_stride]
    x_grid = (np.arange(0, nx, x_stride) * (length / nx)).astype(np.float32)

    return {
        "times": times_ds.tolist(),
        "x": x_grid.tolist(),
        "k": k_phys.tolist(),
        "ic_ids": case_ids.tolist(),
        "eta_truth": truth_eta_ds.round(6).tolist(),
        "eta_pred": pred_eta_ds.round(6).tolist(),
        "err_power": np.log10(np.maximum(err_power_ds, 1e-30)).round(4).tolist(),
        "truth_power": np.log10(np.maximum(truth_power_ds, 1e-30)).round(4).tolist(),
    }


def write_index_html(regimes: list[str], out_path: Path) -> None:
    html = """<!doctype html>
<html lang=en>
<meta charset=utf-8>
<title>DNO/FNO rollout viewer</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 16px; }
  .controls { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
  label { display: flex; flex-direction: column; font-size: 13px; gap: 4px; }
  select, input[type=range] { font: inherit; }
  .plots { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  #status { font-size: 12px; color: #666; }
</style>
<body>
<h1 style="margin:0 0 4px 0;">Rollout viewer</h1>
<div id=status>loading…</div>
<div class=controls>
  <label>Regime
    <select id=regime></select>
  </label>
  <label>IC
    <select id=ic></select>
  </label>
  <label>Time (t)
    <input type=range id=tslider min=0 max=0 step=1 value=0 style="width: 360px;">
  </label>
  <span id=tlabel></span>
</div>
<div class=plots>
  <div id=eta style="height: 360px;"></div>
  <div id=spec style="height: 360px;"></div>
</div>
<div id=loss_over_t style="height: 220px; margin-top: 8px;"></div>

<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script>
const REGIMES = __REGIMES__;
const cache = {};
let cur = null;  // current regime payload

const regimeSel = document.getElementById('regime');
const icSel = document.getElementById('ic');
const tSlider = document.getElementById('tslider');
const tLabel = document.getElementById('tlabel');
const status = document.getElementById('status');

for (const r of REGIMES) {
  const opt = document.createElement('option');
  opt.value = r; opt.textContent = r;
  regimeSel.appendChild(opt);
}

async function loadRegime(name) {
  if (cache[name]) { cur = cache[name]; return; }
  status.textContent = `loading ${name}…`;
  const r = await fetch(`data/${name}.json`);
  cur = await r.json();
  cache[name] = cur;
  status.textContent = '';
}

function relErr(arr_err, arr_truth) {
  return arr_err.map((v, i) => v - arr_truth[i]);  // log10(err) - log10(truth)
}

function render() {
  if (!cur) return;
  const ic = +icSel.value;
  const ti = +tSlider.value;
  const t = cur.times[ti];
  tLabel.textContent = `t=${t.toFixed(2)}`;

  const truth = cur.eta_truth[ti][ic];
  const pred = cur.eta_pred[ti][ic];

  Plotly.react('eta', [
    { x: cur.x, y: truth, name: 'truth', mode: 'lines', line: {color: '#222'} },
    { x: cur.x, y: pred,  name: 'pred',  mode: 'lines', line: {color: '#d33'} },
  ], {
    title: `η(x) at t=${t.toFixed(2)}`,
    xaxis: { title: 'x' },
    yaxis: { title: 'η' },
    margin: {t: 36, b: 40, l: 50, r: 10}, legend: {orientation: 'h'},
  }, {displayModeBar: false});

  const errLog = cur.err_power[ti][ic];
  const truthLog = cur.truth_power[ti][ic];

  Plotly.react('spec', [
    { x: cur.k, y: errLog,   name: 'log₁₀|err|²',   mode: 'lines', line: {color: '#d33'} },
    { x: cur.k, y: truthLog, name: 'log₁₀|truth|²', mode: 'lines', line: {color: '#222'} },
  ], {
    title: `error spectrum at t=${t.toFixed(2)}`,
    xaxis: { title: 'k', type: 'log' },
    yaxis: { title: 'log₁₀ power' },
    margin: {t: 36, b: 40, l: 50, r: 10}, legend: {orientation: 'h'},
  }, {displayModeBar: false});

  // Time history: ‖err‖² summed over k at this IC
  const totals = cur.err_power.map(timeSlice => {
    const ic_arr = timeSlice[ic];
    let s = 0;
    for (let i = 0; i < ic_arr.length; i++) s += Math.pow(10, ic_arr[i]);
    return Math.log10(Math.max(s, 1e-30));
  });
  Plotly.react('loss_over_t', [
    { x: cur.times, y: totals, mode: 'lines', line: {color: '#06c'} },
    { x: [t, t], y: [Math.min(...totals), Math.max(...totals)], mode: 'lines', line: {color: '#d33', dash: 'dot'}, showlegend: false },
  ], {
    title: 'log₁₀ Σₖ |err(k)|² over t (current IC; red = slider position)',
    xaxis: { title: 't' }, yaxis: { title: '' },
    margin: {t: 30, b: 40, l: 50, r: 10}, showlegend: false,
  }, {displayModeBar: false});
}

async function refreshRegime() {
  await loadRegime(regimeSel.value);
  // populate IC selector
  icSel.innerHTML = '';
  for (let i = 0; i < cur.ic_ids.length; i++) {
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = `IC ${i} (case=${cur.ic_ids[i]})`;
    icSel.appendChild(opt);
  }
  tSlider.max = cur.times.length - 1;
  tSlider.value = cur.times.length - 1;
  render();
}

regimeSel.addEventListener('change', refreshRegime);
icSel.addEventListener('change', render);
tSlider.addEventListener('input', render);
refreshRegime();
</script>
</body>
</html>
"""
    html = html.replace("__REGIMES__", json.dumps(regimes))
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        default="outputs/fno_w128b6_v3_hclip5_20260514_063811",
    )
    parser.add_argument(
        "--regimes",
        default="tanaka_g0,tanaka_g1,bf_g0,bf_g1,bf_modal,linear,stokes_deep,stokes_finite,random_sea_deep,random_sea_finite",
    )
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--t_stride", type=int, default=4)
    parser.add_argument("--x_stride", type=int, default=4)
    parser.add_argument("--max_ic", type=int, default=16)
    parser.add_argument("--output_dir", default="playground/rollout_dashboard")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = (repo_root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    eval_dir = run_dir / "eval_suite"
    output_dir = (repo_root / args.output_dir).resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for regime in (r.strip() for r in args.regimes.split(",") if r.strip()):
        npz = eval_dir / f"{regime}_trajs.npz"
        if not npz.exists():
            print(f"skip {regime}: no trajs.npz")
            continue
        payload = build_regime_payload(npz, args.length, args.t_stride, args.x_stride, args.max_ic)
        path = data_dir / f"{regime}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        sz = path.stat().st_size / 1e6
        print(f"  wrote {path}  ({sz:.2f} MB)")
        written.append(regime)

    write_index_html(written, output_dir / "index.html")
    print(f"  wrote {output_dir / 'index.html'}")
    print(
        "\npreview locally:  cd playground/rollout_dashboard && python -m http.server 8000"
        "\ndeploy:           netlify deploy --dir=playground/rollout_dashboard"
    )


if __name__ == "__main__":
    main()
