# Positional identity mechanism study

## Question

Tabular columns have no canonical ordinal order, but a transformer still needs
to route information between simultaneously present feature tokens.  This
study tests whether positional encodings are used as semantic order, as a
content-free internal address, or as a mixture of both.

The key symmetry is explicit.  Without a positional signal, a feature-token
transformer is permutation equivariant: permuting its input feature tokens
permutes the corresponding internal feature-token outputs.  An invariant
readout can therefore remain insensitive to column order.  This desirable
property does not guarantee that a finite model can assign stable computational
roles to otherwise exchangeable or nearly identical feature tokens.  A
positional signal may break that symmetry and become an address used for
routing, even when its ordinal meaning is scientifically irrelevant.

Post-training removal and training without a positional signal answer different
questions.  A model trained on `content + position` may organize later weights
around the position component; deleting it at inference is then a distribution
shift.  A model trained without it can instead learn content-derived routing.

This yields two distinct, falsifiable explanations for a performance drop.
First, inference-only removal can fail because the trained network receives an
activation distribution it never saw during training.  Second, end-to-end
No-PE training can be harder because content-equivalent tokens initially lack a
stable address for specialized computation.  Permutation equivariance is a
desirable symmetry, but it does not by itself guarantee easy optimization or
efficient routing in a finite network.

## Scope and evidence classes

The first mechanism pass compares matched-step exploratory TabICLv2 Stable
RoPE and No-PE checkpoints and the released TabPFN v2.6 model.  Pilot
checkpoints are diagnostic inputs, not confirmatory three-arm evidence.  The
formal `rope`/`temporary`/`none` study remains a separate fresh pre-training
protocol.

Real TALENT and TabArena datasets provide scientific evidence.  Synthetic
linear, XOR, duplicate-column, irrelevant-column, and missing-value tables are
limited to implementation tests.  Input-table perturbation is not the main
experimental method; interventions occur inside the model.

## Pre-registered predictions

1. If positional information is an internal address, feature or group identity
   should be decodable from intermediate activations and on-manifold removal of
   the corresponding directions should change routing or predictions.
2. If the address is needed only to initialize routing, early blocks should be
   more causally sensitive than late blocks and later representations should
   remain recoverable after early activation patching.
3. A No-PE model should rely more strongly on content-derived descriptors such
   as distribution shape, missingness, uniqueness, and target association.
4. Association alone is insufficient.  A feature is called mechanistic only
   when a held-out causal edit changes predictions beyond a frequency- and
   norm-matched random edit, and a paired reverse patch produces a rescue.

Restoring an edited latent to the exact value from the same forward pass is a
round-trip instrumentation control, not a rescue result.  A scientific rescue
must use an independently bound paired source and must have been specified
before held-out evaluation.

## Model-specific interventions

For TabICLv2, RowInteraction RoPE acts over feature-group tokens inside each
row, not over data rows.  Every activation record must retain the mapping from
a group token to its source columns.  Tests cover individual RowInteraction
blocks, query versus key rotation, phase strength, frequency bands, attention
heads, four summary slots, attention/feed-forward branches, and downstream ICL
blocks.

For TabPFN v2.6, the additive feature positional term is decomposed exactly as
`W p + b`.  Conditions retain both terms, retain only `W p`, retain only `b`,
or retain neither.  Feature-attention, row-attention, feed-forward, and selected
residual-stream sites are examined independently.

Zero replacement is never the sole causal control.  It can put activations far
outside the model's training distribution, especially around normalization
layers.  Primary controls use no-op reconstruction, held-out mean replacement,
paired activation replacement, and matched random feature edits.

## Representation analysis

Method-level ablations first select at most two stable causal sites per model.
At each selected site, principal components and a dense autoencoder establish
compression baselines.  An overcomplete top-k sparse autoencoder then uses an
8x dictionary with `k` in `{16, 32, 64}` and training seeds `{42, 43, 44}`.

One dictionary is trained on a balanced mixture of aligned conditions at a
site so feature indices are directly comparable.  Dictionaries are never
shared between TabICLv2 and TabPFN.  A sparse model is interpretable only if it
explains at least 95% of held-out activation variance and reconstruction alone
reduces the native benchmark score by no more than 0.5 percentage points.

Alignment is verified, not inferred from equal array shapes.  Across conditions,
the official inference contract, dataset and sample roster, ensemble views,
feature and class shuffles, feature-group mapping, activation axes, and sampled
coordinates must match.  Each condition retains its own checkpoint digest;
the representation artifact records the complete condition-to-checkpoint and
parent-manifest lineage.

## Data and decisions

TALENT datasets are split by dataset identity into discovery, validation, and
held-out groups using a fixed 70/15/15 manifest after removing TabArena
overlap.  Preprocessing is fitted on training rows only.  The final external
test covers all 38 supported TabArena classification datasets with one fixed
official split per dataset; it is not a leaderboard reproduction.

All comparisons are paired within dataset and report 95% confidence intervals.
Multiple sparse-feature searches control the false-discovery rate.  A proposed
inference intervention is frozen before the held-out and TabArena evaluations.
For held-out runs, latent baselines, target and control features, representation
digest, and sample roster are supplied by a previously hashed freeze artifact.
They are never selected or recomputed from held-out activations.
Content hashes prove what was frozen, not when it was frozen.  A confirmatory
claim additionally requires the freeze artifact to be committed and pushed, or
recorded by another immutable timestamped service, before held-out execution.

If causal address features are beneficial, the follow-up is a
permutation-equivariant content fingerprint rather than a fixed column index.
If positional directions are harmful out of distribution, the follow-up is a
small feature gate.  If computation is concentrated in a few heads, blocks, or
sparse directions, the follow-up is an efficiency ablation.  If no robust
causal feature is found, the result is reported as negative and no large
pre-training run is launched.

## Research anchors

The symmetry argument follows the permutation-invariant and
permutation-equivariant set-function perspective developed by [Deep
Sets](https://arxiv.org/abs/1703.06114) and [Set
Transformer](https://proceedings.mlr.press/v97/lee19d.html).  The intervention
design is grounded in the observation that transformer computations can be
permutation equivariant even when learned positional signals break that
symmetry, as analyzed in [Permutation Equivariance of
Transformers](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Permutation_Equivariance_of_Transformers_and_Its_Applications_CVPR_2024_paper.html).

The model context comes from [TabICLv2](https://arxiv.org/abs/2602.11139), its
[official implementation](https://github.com/soda-inria/tabicl), and the
[TabPFN study](https://www.nature.com/articles/s41586-024-08328-6).  Sparse
autoencoders are used only as candidate feature dictionaries, following the
scaling and evaluation concerns in [Scaling and evaluating sparse
autoencoders](https://arxiv.org/abs/2406.04093); reconstruction quality alone
does not establish a causal mechanism.

## Artifact policy

The public repository contains source, tests, protocol documents, aggregate
statistics, and sanitized provenance only.  Checkpoints, activation shards,
raw predictions, scheduler metadata, logs, telemetry, credentials, and
machine-specific paths remain outside the source tree.  Activation generation
stops below a 20 GiB free-space reserve and is capped at 30 GiB per study.
