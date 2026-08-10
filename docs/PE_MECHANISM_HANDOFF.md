# PE mechanism analysis handoff

**As of:** 2026-08-10 (`Europe/London`)

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
Matched localization first identified a small block-1 loss signal with block 0
as the site control. Corrected discovery and fidelity collections then supported
shared PCA, dense-autoencoder and Top-K sparse-autoencoder fits at both sites.
Only PCA passed the frozen 95% pooled and per-condition fidelity gate. Across
32 discovery-only live interventions, the shift-ranked PCA components were not
consistently more damaging than matched random components, and condition
specificity was unstable. Preserve this as a credible exploratory negative
result; do not promote it to a held-out mechanism or enhancement claim.

The completed whole-row execution used pushed commit
`b2db3517e3dbc9d224daa1651a5ee4e56d430e14`. It captures the complete
512-dimensional `row_interactor` output after internal batching has been
reassembled, which is the supported independently sourced donor boundary.
Before any whole-row collection transaction published output, the exact model
and sampling choices were frozen in
`analysis/pe_mechanism/protocols/whole-row-representation-v2.json`. A released
TabPFN v2.6 checkpoint is still unavailable, so real-model TabPFN experiments
remain gated on official license/access and content verification.

The four whole-row collection transactions and all three pre-registered
representation fits have completed and independently verified.  The discovery
pair contains 111,616 aligned finite vectors per condition.  PCA-512 passed
with validation explained variance 1.0 and dense AE-384 passed with 0.9773
pooled, 0.9740 No-PE and 0.9727 RoPE.  The Top-K SAE failed decisively at
0.6034 pooled validation explained variance despite 0.9906 training fidelity;
do not tune it on the fixed fidelity roster or carry it into causal edits.
PCA passed the recipient and source live no-op gates on the first causal table
before the unequal-dose gate stopped the run, but those gates remain mandatory
per table. Dense AE has not undergone a live model-causal run.

The frozen 73-table whole-row PCA ranking selected targets `[0, 7, 5, 3]`
and matched controls `[181, 502, 153, 310]`. Its first live causal attempt
published no intervention prediction: the first table's decoded ablation dose
ratio was `7.81832`, above the frozen `1.25` gate, so the transaction failed
closed with no output directory. Do not treat that failed attempt as a model
effect or reuse the unequal full edits.

`whole-row-causal-dose-amendment-v1.json` freezes the outcome-free repair. It
binds the original split and ranking, matches target/control decoded RMS for
each raw call by preserving the smaller full edit byte-for-byte and shrinking
only the larger latent edit, applies the rule separately to ablation and donor
patches, and rejects any actual per-side dose increase before prediction. Zero
or non-finite doses fail closed. Report
the resulting operations as partial edits and never compare ablation and donor
effect magnitudes as though those two families share one common dose.

Exact analysis SHA `fd3b1d9addafc46533d1cb461e347f46ab2c8c96` passed the
complete software/public gates, regenerated the identical target/control
ranking with the dose amendment bound, and completed the one permitted
eight-table RoPE-source to No-PE-recipient campaign on a non-H100 accelerator.
All execution gates passed. The maximum full-edit decoded-dose ratio was 37.63
and the maximum executed ratio was 1.000000064 with no amplification.

The result is a credible exploratory negative. Ablation target-minus-control
mean log loss was `-7.83e-4` (3/8 positive; descriptive exact `p=0.2422`).
Donor target-minus-control mean log-loss improvement was `1.56e-4` (4/8
positive; descriptive `p=0.5547`). Preserve the sanitized finding at
`analysis/pe_mechanism/findings/whole-row-pca-dose-matched-exploratory-negative-v1.json`.
Do not run the reverse direction or dense AE merely to search for a positive
result; the stopping rule has fired.

The same legacy matched-step pair has completed a TabArena 38-task lite
evaluation against the released reference model. Mean rank (lower is better)
was `1.342` for released, `2.184` for No-PE, and `2.474` for RoPE, with 28, 6,
and 4 task wins respectively. No-PE beat RoPE on 24/38 tasks, but the paired
exact result (`p=0.1433`) does not establish a reliable advantage. This run is
explicitly `formal_eligible=false` and is not a leaderboard reproduction. Do
not promote it to formal evidence or rerun it merely to search for significance.

The descendant package now also contains a separate formal three-arm
checkpoint-intake validator and a complete Stage-3 example contract. It checks
the canonical RoPE/Temporary/No-PE chains, exact checkpoint and finalization
bytes, the shared transaction ledger, exact training checkout, cross-arm
cohort provenance, and path-free reporting. It intentionally stops before
benchmark execution and reports campaign acceptance, terminal scheduler
evidence, and benchmark readiness as false. Do not turn those fields true from
the checkpoint ledger alone; they require independently verified production
acceptance artifacts.

The offline TabPFN v2.6 positional-decomposition wrapper has passed its
software tests. No real v2.6 run exists because the official weight is not
available to this workstream; license acceptance, authenticated access, and
content verification remain required.

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

1. Preserve the verified whole-row artifacts and representation decision:
   PCA and dense AE qualify; Top-K SAE is rejected. Do not recollect
   activations or retrain these dictionaries.
2. Preserve the dose-amendment execution at exact SHA `fd3b1d9...`, its
   unchanged target/control ranking and the completed eight-table aggregate.
   Do not rerun or expand the whole-row PCA branch: it met the pre-declared
   stopping condition for a credible exploratory negative.
3. Do not run the reverse donor direction or dense AE on these legacy
   checkpoints merely to search for a favorable result. Reopen either only for
   a separately justified, outcome-independent question.
4. Keep the sanitized finding and private aggregate content-bound; never
   publish the raw predictions, checkpoint material, logs or machine paths.
5. Treat the 36-table result as exploratory because the unsupervised
   representation dictionary was trained across all 109 discovery datasets;
   it is outcome-disjoint, not representation-level untouched.
6. Preserve the completed 38-task lite TabArena aggregate as exploratory only;
   it is neither formal-eligible nor a leaderboard reproduction and must not be
   used as a substitute for matched three-arm confirmation.
7. Do not resume broad block-0/1 collection over large tables until capture
   metadata explicitly represents repeated internal chunk invocations. The
   completed small-roster block-level PCA negative remains valid.
8. With matched formal checkpoints, run model interventions on every frozen
   validation dataset, including dose-matched controls and the independently
   bound paired reverse patch. Do not use the word rescue unless all
   source-native/source-no-op and dose gates pass.
9. First pass the formal checkpoint-intake validator, then independently bind
   campaign acceptance and terminal scheduler evidence before any formal
   TALENT/TabArena execution.
10. Acquire and content-bind the released TabPFN v2.6 checkpoint, then run the
   registered `W p + b`, `W p`-only, `b`-only, and neither decomposition.
11. Hash and externally pre-register the selected intervention, baseline,
   controls, validation fingerprints, and held-out sample rosters. Then run the
   complete candidate-by-held-out-dataset Cartesian set and apply the frozen
   six-component intersection-union plus Holm confirmation rule.
12. Publish only sanitized code, protocol, aggregate statistics, and provenance.
