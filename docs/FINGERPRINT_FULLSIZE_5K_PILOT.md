# Full-size RoPE–Fingerprint 5k Pilot

This is a paired exploratory Stage-1 screening run, not part of the formal
three-arm position–identity campaign.

Both arms start from scratch with seed 42 and use the released-size TabICLv2
backbone: embedding dimension 128; three column blocks; three row blocks; and
twelve ICL blocks. They share the GraphSCM prior, batch construction,
optimizer, scheduler, training precision, sequence limits, and 5,000-update
budget.

The only intended treatment difference is:

| Arm | `row_identity_mode` | `row_fingerprint` | fingerprint dimension |
| --- | --- | --- | --- |
| RoPE | `rope` | `False` | inactive |
| Fingerprint | `none` | `True` | 16 |

The learned fingerprint injection adds 24,582 trainable parameters (about
0.089%) to the otherwise identical full-size backbone. This is recorded as an
induced treatment difference, not hidden as parameter equality.

The official Stage-1 run warms up for 1% of 500,000 updates, or 5,000 updates.
This bounded pilot therefore uses a fixed 5,000-step warmup even though it
stops at update 5,000. Its learning-rate values through the observed prefix
match the official warmup prefix. The resulting checkpoint remains
exploratory and is not eligible for formal resume or formal evidence.

The submission controller creates a fresh external namespace, submits the two
one-H100 allocations held, and releases them only after both submissions
succeed. Any partial held submission is cancelled. Each job validates the
exact clean source SHA, H100 allocation, final step, architecture, treatment,
checkpoint size, and checkpoint SHA-256. OOM and non-finite losses fail the
run. Checkpoints are written every 1,000 updates with at most two retained per
arm; there are no Stage-2 or Stage-3 dependencies.
