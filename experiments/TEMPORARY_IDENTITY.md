# Position vs. Temporary Identity

This branch supports pre-training-aligned tests of the feature-position signal
in TabICLv2's row interaction transformer. The signal is applied across the
feature-token sequence inside each table row; it is not a sample-row position.

## Matched training arms

| `--row_identity_mode` | RoPE | Feature assignment | Interpretation |
| --- | --- | --- | --- |
| `rope` | yes | input order | standard TabICLv2 baseline |
| `temporary` | yes | fresh random assignment per table and forward | identity without stable ordinal meaning |
| `none` | no | set aggregation only | no explicit symmetry-breaking identity |

Temporary assignments are shared across every sample row in a table. CLS
tokens remain fixed, and padding masks are permuted with their feature tokens.
The transform is otherwise the same parameter-free, norm-preserving RoPE used
by the baseline.

All three arms must be trained from scratch with identical prior settings,
optimizer, architecture, step budget, and seeds. Do not compare a modified
inference path against an unmodified pre-trained checkpoint as primary evidence.

## Reproducibility gates

1. Unit tests:

   ```bash
   .venv/bin/python -m pytest -q tests/test_row_identity.py
   ```

2. One-step end-to-end smoke tests on one GPU:

   ```bash
   for mode in rope temporary none; do
     CUDA_VISIBLE_DEVICES=3 scripts/run_identity_smoke.sh "$mode"
   done
   ```

3. Run sequential 500-step `none` preflights with 1, 2, and 4 H100s while
   keeping the global batch at 64. Each job records only its allocated GPUs
   every five seconds and checks every GPU for at least 80% mean utilization:

   ```bash
   scripts/submit_h100_scaling_preflights.sh
   ```

4. Select the largest GPU count for which every allocated card passes. Submit
   matched `rope` and `none` full runs with that count. Each arm runs the
   official 500k + 40k + 10k stages; downstream stages start only after the
   preceding checkpoint and utilization gate pass:

   ```bash
   scripts/submit_h100_rope_none_full.sh GPU_COUNT
   ```

The full jobs use FP32 to match the official recipe. Stages 2 and 3 require
FlashAttention-3 on Hopper; Stage 3 enables activation recomputation for H100
memory headroom. On-the-fly prior workers are capped per DDP rank, and the CUDA
caching allocator is retained between steps to prevent avoidable GPU stalls.

## Required evaluation matrix

Evaluate every checkpoint on the same datasets and preprocessing under:

- canonical feature order;
- multiple random feature permutations (report mean and worst case);
- train/test-consistent renaming of feature identities;
- feature-count buckets, including counts outside the pilot's common range;
- in-prior synthetic tasks and held-out real classification tasks;
- calibration and log loss in addition to rank/accuracy metrics.

For `temporary`, average multiple independently sampled identities at inference
and plot performance against ensemble size. This distinguishes a useful
temporary binding mechanism from variance caused by a single random draw.

## Current execution scope

The first full allocation covers `rope` and `none`. `temporary` remains tested
and checkpoint-compatible but is not submitted at full scale until the first
matched comparison is reviewed.
