# PE mechanism analysis handoff

## Stable contract

This workstream is a descendant of the immutable formal training candidate at
`f06c1f130397ade4605ab9ce6769a363803d02e5`.  It must not modify or weaken the
formal three-arm training protocol.  Mechanism and evaluation commits are
recorded separately from the training commit in every result manifest.

The first pass studies two models:

- matched-step exploratory TabICLv2 Stable RoPE and No-PE checkpoints;
- the released TabPFN v2.6 model under exact internal positional-term
  decomposition.

Temporary identity remains part of the confirmatory training study but is not
included in the first representation comparison without a reliable matched
checkpoint.

The analysis package is a descendant workstream. It must never be imported by
or copied into a running exploratory training job. Every evidence run uses a
clean, detached checkout of the exact training/model SHA and records the exact
analysis SHA separately.

## Scientific invariants

- Real datasets provide evidence; synthetic datasets validate tools only.
- Internal interventions, not artificial table perturbations, are the primary
  method.
- TabICLv2 RowInteraction RoPE indexes feature-group tokens inside each row.
  Activation records must preserve the group-to-source-column map.
- TabPFN v2.6 positional input is separated into `W p`, bias `b`, both, and
  neither.
- Zero replacement is not a sufficient causal control.  Use reconstruction,
  held-out means, paired patches, and matched random features.
- Association is not called a mechanism without held-out causal change and a
  paired rescue.
- Restoring a latent from the same forward pass is a round-trip implementation
  control, not a paired rescue.
- Dictionaries are shared across aligned conditions at one site, never across
  TabICLv2 and TabPFN.
- Shared dictionaries retain a separate checkpoint and parent-manifest digest
  for every condition. Matching shapes do not prove activation alignment;
  official view, shuffle, grouping, axis, roster, and coordinate evidence must
  match.

## Data contract

TALENT discovery, validation, and held-out sets are assigned by dataset
identity using a committed deterministic manifest after removing TabArena
overlap.  Preprocessing is fitted on training rows only.  The external test
uses all supported TabArena classification datasets with one fixed official
split per dataset.

The sparse-autoencoder qualification thresholds are fixed before held-out
evaluation: at least 95% held-out activation variance explained and no more
than 0.5 percentage points of native-score loss from reconstruction alone.

Discovery and validation may be used to select candidate sparse features and
matched controls. Held-out runs require a previously hashed freeze artifact
with the numerical baseline, target/control features, representation digest,
and sample roster. No held-out activation may be used to recompute those
choices.

Formal paired work is restricted to `row_interactor`. Feature identifiers are
coordinates in a frozen representation of the flattened RowInteraction CLS
slots; they are not raw columns, row positions, attention heads, positional
indices, or single neurons. An `exploratory_pilot` checkpoint study may be used
only for discovery and cannot enter validation, selection, or held-out
confirmation.

## Artifact boundary

Source, tests, protocol documents, sanitized manifests, and aggregate
statistics may be public.  Checkpoints, activation shards, raw predictions,
logs, telemetry, scheduler metadata, credentials, personal information, and
machine-specific paths must remain outside the source tree and public Git
history.

Generated outputs require an absolute directory outside this checkout.  The
study caps private activations at 30 GiB and stops new generation below a
20 GiB free-space reserve.

## Ordered continuation

1. Finish and content-verify the step-210000 Stable RoPE snapshot; the No-PE
   snapshot at that step is already verified.
2. From clean exact SHAs, run official collection on small discovery and
   validation datasets and prove the complete collect-to-representation-to-live
   intervention path.
3. Run method-level ablations on the TALENT discovery roster and propose
   candidate latent features. The implemented validation workflow may freeze
   at most two candidates using candidate-level six-component tests and
   Benjamini-Yekutieli correction; do not select sites or features from
   held-out data.
4. Train PCA, dense, and sparse representations on aligned Stable RoPE/No-PE
   activations. Reject any coordinate or inference-contract mismatch.
5. Run model interventions on every frozen validation dataset, including
   dose-matched controls and the independently bound paired reverse patch. Do
   not use the word rescue unless all source-native/source-no-op and dose gates
   pass.
6. Hash and externally pre-register the selected intervention, baseline,
   controls, validation fingerprints, and held-out sample rosters. Then run the
   complete candidate-by-held-out-dataset Cartesian set and apply the frozen
   six-component intersection-union plus Holm confirmation rule.
7. Publish only sanitized code, protocol, aggregate statistics, and provenance.
