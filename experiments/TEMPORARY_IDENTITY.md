# Do Tabular Foundation Models Need Position, or Just Temporary Identity?

This experiment tests the feature-position signal in TabICLv2's row
interaction transformer. “Row” here names the architectural block; the token
sequence being treated is the table's feature sequence, not the order of data
samples.

## Matched pre-training arms

| `row_identity_mode` | Treatment | Interpretation |
| --- | --- | --- |
| `rope` | Standard RoPE follows input feature order | Stable ordinal position |
| `temporary` | RoPE identities are randomly reassigned per table/forward pass | Temporary identity without stable order |
| `none` | Row RoPE is disabled | No explicit positional identity |

Temporary assignments are shared across the sample rows of a table. CLS tokens
remain fixed, and padding masks move with their feature tokens. The assignment
stream has an independent, checkpointed RNG so resume does not perturb or
depend on the global training RNG.

All three arms are trained from scratch with the same source, prior,
architecture, optimizer, stage settings, budgets, and non-treatment seeds. The
fixed stage budgets are 500,000, 40,000, and 10,000 steps. A shared cohort hash
excludes only `row_identity_mode`; a separate arm hash binds the corresponding
identity treatment. Inference-only removal of position from an otherwise
pretrained checkpoint is not primary evidence for this study.

## Formal evidence boundary

The public candidate contains the training treatment, exact stochastic resume,
strict checkpoint/parent provenance, capacity and retention gates, the fixed
twelve-case H100 validation harness, a fresh-only nine-job submit transaction,
and a read-only anomaly monitor. `docs/FORMAL_STATUS.md` records readiness and
`docs/FORMAL_HANDOFF.md` defines the remaining hardware gates.

At the current status, CPU/Gloo and hermetic scheduler tests are ready, while
H100/CUDA/NCCL/utilization evidence is pending. No result from the formal
three-arm training comparison is claimed yet.

## Evaluation plan (separate candidate)

After all three formal training arms finish, evaluate their terminal
checkpoints under one frozen downstream protocol:

- canonical feature order and repeated random feature permutations;
- train/test-consistent feature renaming;
- feature-count buckets, including out-of-prior counts;
- in-prior synthetic tasks and held-out real classification tasks;
- calibration and log loss in addition to rank and accuracy metrics;
- for `temporary`, multiple independently sampled identities at inference and
  performance versus ensemble size.

Evaluation implementation, datasets, raw predictions, and result artifacts are
deliberately outside the formal training candidate so they cannot change its
source identity.
