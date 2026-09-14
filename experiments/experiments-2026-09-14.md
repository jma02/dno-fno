| Date | Time | File / Variant | Motivation | What Tried / Evidence | Correctness | Timing | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-14 | 01:50 | Mixed-wave fine-tune: completed training and partial128-case rollout panel | Restoring omitted mixed-wave examples might improve the balanced model's failures without changing architecture, normalization or rollout numerics. User requested completion status. | Five-epoch run c27_balanced_mixed_wave_finetune_20260913 completed September13 at19:39, final/best epoch5. Of128 queued FP64 damping-off rollouts,88 are saved:32Stokes,32JONSWAP,24Tanaka. All Stokes/JONSWAP finite;3of24 Tanaka nonfinite so far. Priority16471 stays finite toT200 but final surface relative-L2 error is.96559049 versus archivedFP32 reference; before fine-tuning it became nonfinite at179.2. Priority16624 now first saves NaNs at27.2 versus150.4 before fine-tuning. Other observed nonfinite cases16494at96.8 and16505at191.2. | Training metrics finite and final checkpoint exists. Priority metadata uses identical IC IDs/depths, FP64 model/integration, internaldt.01, four GL2 iterations, cutoff128, no extra stabilizer, same archivedreference. Full panel incomplete; do not report3/128 as a final failure rate. Finite16471 is not accurate recovery. No new training/evaluation launched for this status check. | 00:55 completed training (3298.9928s). Rollout outputs span September13–14;88/128 saved at01:50, final evaluation timing pending. | Do not adopt this checkpoint as a fix. Packet-only low-LR fine-tuning worsens the timing of16624 and does not accurately recover16471. This does not isolate missing-data effects from packet-only adaptation/forgetting or establish what from-scratch mixed-data training would do. Keep the existing panel unchanged; no automatic new experiment. |

Artifacts:

- Training: `outputs/c27_balanced_mixed_wave_finetune_20260913/{summary.json,train_log.jsonl,final_ckpt}`.
- Results: `outputs/c27_balanced_mixed_wave_finetune_20260913/eval_final_n32_fp64/simulation_*.json/.npz`.
- Comparison: `outputs/c27_balanced_tanaka_tangent_modal_20260913/eval_final_n32_fp64/simulation_{16471,16624}.json/.npz`.

Read-only aggregation uses `all_saved_values_finite` from each case JSON; absent
keys such as `model_nonfinite_any` must not be treated as proof of finiteness.
Independent NPZ review confirms byte-identical initial eta/xi, depths, simulation
IDs and251-frame time grids against the baseline for both priority ICs; all
saved physical fields are float64. Recomputed errors agree within7.9e-13.
The completed families are not a random subset of128: known Tanaka failures ran
first, then Stokes/JONSWAP, then remaining Tanaka. BF has not yet saved results.
