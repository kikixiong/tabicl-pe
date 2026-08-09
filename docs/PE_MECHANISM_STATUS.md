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

The complete current v2 worktree regression is 1080 passed (2 skipped) for the
root package and 323 passed for the mechanism package; Ruff also passes over
the complete mechanism source and test trees. These are implementation checks,
not scientific results. The exact tree must still be committed and pushed
before a scheduled run can satisfy strict analysis-code provenance.

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
step-250000 downstream evaluation and mechanism comparison have not yet been
run. Their discovery-only runner and H100 Slurm wrapper are now implemented,
but they do not become immutable evidence until this branch is committed,
pushed, and executed from a clean exact checkout. A real released TabPFN v2.6
checkpoint is not locally available to this workstream; its current coverage is
the offline official driver and synthetic API tests, not a real-model result.

Still required before a mechanistic claim:

- matched step-250000 downstream and mechanism runs from the verified Stable
  RoPE/No-PE exploratory pair;
- real independently sourced paired reverse-patch runs over the frozen
  validation roster. Restoring the same latent is only a round-trip plumbing
  control;
- a pushed immutable descendant analysis SHA with its CPU regression and public
  hygiene checks repeated; the immutable training SHA and its 12/12 H100 gate
  are already established;
- actual validation selection, an externally pre-registered freeze, and the
  complete held-out TALENT confirmation run before any feature is called
  replicated.

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
