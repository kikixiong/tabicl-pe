# PE mechanism analysis status

**As of:** 2026-08-07 (`Europe/London`)

## Current phase

Implementation is in final integration on descendant branch
`codex/pe-mechanism-sae-v1`, based on public formal candidate `f06c1f1`.
The formal training candidate and the running exploratory training scripts
remain unchanged.

Implemented and regression tested:

- official train-only TabICLv2 activation capture through the public sklearn
  preprocessing and ensemble path;
- exact PCA, dense autoencoder, and top-k sparse-autoencoder baselines;
- strict parent-run, checkpoint, dataset, code, activation-coordinate, and
  condition lineage;
- official model-in-the-loop no-op, target, matched-control, and round-trip
  interventions;
- deterministic TALENT/TabArena dataset manifests, paired statistics, atomic
  outputs, storage gates, and public-safe provenance;
- version-specific TabICLv2 and TabPFN v2.6 instrumentation tests.

The No-PE exploratory checkpoint at exactly step 210000 has been copied and
content-verified outside the repository. A real strict smoke run bound analysis
commit `1c0f665dfc92bf6c0e49d9a0e8e57e9619d0ef9c` and model/training commit
`2d44e7540ffd4ec6216aa0058970b1de1034aa1d` to the complete official path:

- official discovery and validation collection produced separate bounded
  `1024 x 128` activation shards with explicit four-token CLS offsets;
- a full-SVD 128-component PCA reconstructed validation activations with
  explained variance `1.0`;
- live official model intervention passed all no-op gates, including maximum
  probability deviation `5.96e-7` and zero measured accuracy change.

This verifies plumbing and provenance, not a positional mechanism. The
corresponding Stable RoPE checkpoint has not yet reached step 210000, so no
matched-condition mechanism comparison has been run. A real released TabPFN
v2.6 checkpoint is not locally available to this workstream; its current
coverage is adapter validation only.

Still required before a mechanistic claim:

- a content-verified matched Stable RoPE checkpoint at the same step;
- an independently sourced paired reverse-patch rescue. Restoring the same
  latent is only a round-trip plumbing control;
- a verified multi-dataset selection workflow that applies false-discovery-rate
  control before freezing candidate features;
- frozen feature/control selection before held-out TALENT or TabArena use.

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
