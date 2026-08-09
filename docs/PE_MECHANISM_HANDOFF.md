# PE mechanism analysis handoff

## Stable contract

This workstream is a descendant of the immutable formal training candidate at
`23f476d20c3fda898b547a19d0918bb0d72438b4`. It must not modify or weaken the
formal three-arm training protocol.  Mechanism and evaluation commits are
recorded separately from the training commit in every result manifest.

The exact training candidate has passed its canonical twelve-case H100 matrix
12/12. Formal three-arm production training has not been submitted. The H100
result establishes training-candidate readiness only; it does not establish a
positional mechanism or validate the descendant analysis code.

The first pass studies two models:

- content-verified step-250000 exploratory TabICLv2 Stable RoPE and No-PE
  checkpoints;
- the released TabPFN v2.6 model under exact internal positional-term
  decomposition.

Temporary identity remains part of the confirmatory training study but is not
included in the first representation comparison without a reliable matched
checkpoint.

The step-250000 pair belongs to the legacy pilot lineage and is discovery-only.
Matched step-250000 localization has now been executed from pushed analysis
commit `4b797135b655ee181647c84fed3df946f942f248`. The result localizes the strongest
small loss effect to RowInteraction block 1 and uses block 0 as the site
comparison; it does not establish a mechanism or an enhancement. Strict paired
activation collection for both sites and both arms has also completed on eight
discovery datasets. The first independent eight-dataset fidelity collection is
quarantined: it exposed a mixed-case site-roster ordering bug and a
process-hash-dependent normalization-view order, so strict representation
alignment rejects it. Re-collect that roster only after the canonical ordering
fix and hash-seed-frozen wrapper are pushed. A released TabPFN v2.6 checkpoint
is not currently available locally, so its real-model experiments cannot begin
until official license/access requirements are satisfied and the checkpoint is
content-bound.

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

1. Commit and push the canonical mixed-case site-roster fix and the
   `PYTHONHASHSEED=0` scheduled collector, then re-collect the independent
   fidelity roster under that exact commit. Require the strict parser and every
   cross-condition normalization view, shuffle, call index and sampled
   coordinate to match before continuing. Never manually reorder an existing
   index.
2. Train shared PCA, dense-autoencoder, and top-k sparse-autoencoder
   representations on the completed aligned Stable RoPE/No-PE block-1
   activations, with block 0 as the matched site control. Use the corrected
   validation-assigned activations only for the frozen reconstruction-fidelity
   gate; do not select features or methods from them. The pilot pair must never
   be used for validation or held-out mechanism claims.
3. Run live method-level interventions on the TALENT discovery roster and
   propose
   candidate latent features. The implemented validation workflow may freeze
   at most two candidates using candidate-level six-component tests and
   Benjamini-Yekutieli correction only after matched formal checkpoints exist;
   do not promote the exploratory pilot into that workflow.
4. Reject any representation whose condition coordinates, official inference
   contract, checkpoint lineage, or reconstruction-fidelity gate fails.
5. With matched formal checkpoints, run model interventions on every frozen validation dataset, including
   dose-matched controls and the independently bound paired reverse patch. Do
   not use the word rescue unless all source-native/source-no-op and dose gates
   pass.
6. Acquire and content-bind the released TabPFN v2.6 checkpoint, then run the
   registered `W p + b`, `W p`-only, `b`-only, and neither decomposition.
7. Hash and externally pre-register the selected intervention, baseline,
   controls, validation fingerprints, and held-out sample rosters. Then run the
   complete candidate-by-held-out-dataset Cartesian set and apply the frozen
   six-component intersection-union plus Holm confirmation rule.
8. Publish only sanitized code, protocol, aggregate statistics, and provenance.
