# TabICLv2 task-conditioned fingerprint pilot

This is an exploratory, non-formal experiment. It does not modify or replace the
frozen `position-identity-v1` protocol.

## Question

Can a compact, task-conditioned feature fingerprint recover part of the gap
between No-PE and ordered Row-RoPE without assigning a persistent ordinal
position to a tabular feature?

## Matched arms

- `rope`: ordered Row-RoPE, fingerprint disabled.
- `none`: Row-RoPE disabled, fingerprint disabled.
- `fingerprint`: Row-RoPE disabled; a fingerprint computed only from the
  training rows is routed into row-attention queries and keys.

All three arms start from scratch with seed 42 and share the same synthetic
prior, data order, small architecture, optimizer, schedule, batch budget, and
10,000 Stage-1 optimizer steps. Each array task requests exactly one A10; Slurm
decides when each independent arm starts according to cluster availability.
The screening run uses the same FP16 automatic-mixed-precision setting for all
three arms; it is not a replacement for the release-scale FP32 protocol.

## Fingerprint boundary

The column embedder remains target-aware. For each feature-group token, its
training-row embeddings are averaged once per table. A small projection produces
the compact fingerprint. The same fingerprint is reused for every row and every
row-interaction block. Layer-specific projections add it only to attention Q/K:

- full row-attention blocks: feature-token Q and K;
- final CLS readout block: feature-token K only;
- never V, never the residual stream, and never CLS identity.

Because the existing circular feature grouping remains enabled, this pilot tests
replacement of ordinal **Row-RoPE**. It does not establish full raw-column
permutation invariance.

## Pre-registered causal reads

The trained fingerprint arm supports inference with the correct fingerprint,
zero contribution, or a within-table permutation. Correct beating zero shows
that the routing signal is used. Correct beating permuted shows that binding the
fingerprint to the matching feature content matters; otherwise it behaves more
like an anonymous symmetry breaker.

Exact duplicate feature-group representations are intentionally allowed to have
the same fingerprint.

## Promotion rule

Seed 42 is a screening pilot, not a paper result. It advances only if the
fingerprint arm recovers a meaningful fraction of the measured RoPE-minus-No-PE
gap and the correct/zero/permuted interventions establish the claimed mechanism.
Any promoted result must be repeated with matched seeds and a separate held-out
evaluation protocol.
