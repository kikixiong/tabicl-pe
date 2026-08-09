# PE mechanism analysis status

**As of:** 2026-08-09 (`Europe/London`)

## Current phase

The mechanism package is being carried forward on descendant branch
`codex/pe-mechanism-sae-v2`. Its immutable training ancestor is
`23f476d20c3fda898b547a19d0918bb0d72438b4` on
`codex/position-identity-v1`. The canonical twelve-case H100 validation matrix
has passed 12/12 for that exact training commit. This hardware result qualifies
the frozen training candidate; it is not itself mechanism evidence. Formal
three-arm production training has not been submitted.

Implemented and regression tested:

- official train-only TabICLv2 activation capture through the public sklearn
  preprocessing and ensemble path;
- exact PCA, dense autoencoder, and top-k sparse-autoencoder baselines;
- strict parent-run, checkpoint, dataset, code, activation-coordinate, and
  condition lineage;
- official model-in-the-loop no-op, target, matched-control, and round-trip
  interventions;
- independently bound Temporary-to-RoPE paired reverse patches with native and
  no-op source baselines plus latent and decoded-dose balance gates;
- candidate-level six-component intersection-union tests with
  Benjamini-Yekutieli correction on validation, immutable feature freezing,
  and six-component Holm-confirmed replication on the complete held-out
  candidate-by-dataset set;
- full probability/label/loss recomputation, exact-content and
  invariant-prediction dataset-alias rejection, and strict formal checkpoint
  lineage from Stage 1 through the evaluated stage;
- deterministic TALENT/TabArena dataset manifests, paired statistics, atomic
  outputs, storage gates, and public-safe provenance;
- version-specific TabICLv2 and TabPFN v2.6 instrumentation tests.
- a strict matched-step TabICLv2 localization runner that binds both step-250000
  pilot checkpoints, their common architecture/state schema, every TALENT input,
  the exact PE policies, paired official ensemble schedule, safely encoded labels,
  stored prediction bytes, and runtime versions before atomic publication;
- an offline-only official TabPFN v2.6 driver that fits TALENT train rows once
  and evaluates `W p + b`, `W p`, `b`, and neither on the same fitted model,
  with exact full/native and hook-restoration gates.

The current immutable mechanism execution commit is
`b2db3517e3dbc9d224daa1651a5ee4e56d430e14`, pushed on
`codex/pe-mechanism-sae-v2`. It descends from the first execution commit
`4b797135b655ee181647c84fed3df946f942f248` and adds whole-row
`row_representation` support without changing the frozen training candidate.
Its complete mechanism regression was 338 passed, with Ruff and diff checks
also passing. These are implementation checks, not scientific results.

An exploratory Stable RoPE/No-PE checkpoint pair at exactly step 250000 has
been copied and content-verified outside the repository. Both checkpoints load
successfully, report the same training step, and match their declared identity
modes. They come from the legacy pilot lineage and remain
`formal_eligible=false`; they may support discovery only and cannot be combined
with a future Temporary checkpoint as confirmatory three-arm evidence.

An earlier real strict smoke run on a No-PE pilot checkpoint exercised the
complete official path:

- official discovery and validation collection produced separate bounded
  `1024 x 128` activation shards with explicit four-token CLS offsets;
- a full-SVD 128-component PCA reconstructed validation activations with
  explained variance `1.0`;
- live official model intervention passed all no-op gates, including maximum
  probability deviation `5.96e-7` and zero measured accuracy change.

This verifies plumbing and provenance, not a positional mechanism. The matched
step-250000 discovery localization has now also run from a clean checkout of
the pushed analysis commit on eight TALENT datasets. The run and every stored
prediction were independently re-hashed and the metrics were recomputed from
the stored float32 probabilities. The native RoPE condition was byte-exact to
the official baseline.

The localization result is diagnostic rather than a win/loss claim. Disabling
RoPE only in RowInteraction block 1 produced the clearest repeatable loss
signal: mean log-loss effect versus native was about `-7.55e-4`, where negative
means the intervention was worse. Block 0 was a useful matched comparison and
block 2 was near-neutral. Several PE edits changed individual probabilities
substantially while mean accuracy and log loss moved little. Native No-PE and
RoPE were also close in aggregate on this small roster despite large prediction
differences, so downstream scores alone do not explain the representation
mechanism.

Following that result, official activations were collected for block 1 and
block 0 from both matched checkpoints. The strict runs cover eight discovery
datasets, two sites, and two conditions. Every shard contains 8192 sampled
128-dimensional vectors; paired RoPE/No-PE call indices and activation
coordinates are exactly equal, and independent verification found no
non-finite values or public-metadata path leakage.

The disjoint eight-dataset representation-fidelity roster was recollected with
canonical ordering and a frozen hash seed, and strict cross-condition schedule
and coordinate checks passed. Shared block-0 and block-1 representation models
were then trained on the aligned discovery activations. Full-rank PCA passed
the pooled and per-condition 95% validation explained-variance gate at both
sites. Dense autoencoders reached only about 0.91--0.93, and Top-K sparse
autoencoders about 0.77--0.80, on the disjoint fidelity roster; those nonlinear
models are therefore ineligible for live causal claims.

Thirty-two discovery-only PCA model interventions subsequently passed every
no-op and reconstruction gate. Deleting components ranked by the RoPE/No-PE
activation shift did not consistently hurt predictions more than deleting
matched random components, and the RoPE-minus-No-PE specificity interaction
was also unstable. This is a credible exploratory negative result for those
block-level PCA directions, not held-out confirmation and not evidence that
positional encoding is irrelevant.

The next execution phase uses the complete 512-dimensional output of
`row_interactor`, the only supported boundary for independently sourced
RoPE-to-No-PE and No-PE-to-RoPE patches. Collection covers 109 eligible
discovery datasets and the fixed eight-dataset fidelity roster. Before any
collection transaction published output, the representation choices were
frozen in
`analysis/pe_mechanism/protocols/whole-row-representation-v2.json`: full-rank
PCA-512, dense AE-384 and an 8x Top-K SAE with `k=64`, with deterministic
equal-per-dataset sampling and no tuning on the fidelity roster.

A real released TabPFN v2.6 checkpoint is not locally available to this
workstream. The mutable descendant tree now contains an offline official
fixed-weight TALENT runner for `Wp+b`, `Wp`, `b`, and zero-position conditions,
but its real-model run remains gated on official license acceptance, token-based
checkpoint access, content verification, a new clean pushed analysis commit,
and a clean TabPFN v7.1.1 checkout.

Still required before a mechanistic claim:

- verify every 109-dataset whole-row collection artifact and exact
  cross-condition schedule, view, shuffle and sampled-coordinate alignment;
- train the pre-registered whole-row representations and admit only models that
  pass pooled and per-condition fidelity plus live no-op gates;
- run discovery-only target deletion, matched random controls and independently
  sourced patches in both RoPE/No-PE directions. Restoring the same latent is
  only a round-trip plumbing control;
- acquire and content-bind the licensed released TabPFN v2.6 checkpoint before
  running its registered positional-term decomposition;
- obtain matched formal RoPE, Temporary and No-PE checkpoints before any
  validation selection or held-out TALENT/TabArena confirmation. The legacy
  pair cannot be promoted into that workflow.

No mechanism result, sparse feature, downstream improvement, or efficiency
gain is claimed at this stage.  Existing pilot checkpoints remain diagnostic
and cannot be promoted to confirmatory three-arm evidence.

## Acceptance gates

- Disabled instrumentation reproduces model outputs within `1e-6`.
- TabPFN's separated `W p + b` components numerically reconstruct the original
  positional term.
- Activation axes and feature-group mappings are explicit and tested.
- Representation reconstruction passes the pre-registered fidelity threshold.
- Causal effects replicate on held-out datasets and exceed matched random
  controls after the intervention and controls have been frozen on validation
  evidence.
- A paired rescue uses an independently bound source; a same-pass round trip
  does not satisfy this gate.
- Public-history and artifact hygiene tests pass before every push.
