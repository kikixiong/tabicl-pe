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

The immutable whole-row collection execution commit is
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
The 109 discovery datasets are also frozen into 73 feature-ranking datasets and
36 disjoint causal-test datasets by
`analysis/pe_mechanism/protocols/whole-row-causal-split-v1.json`.  The
representation dictionary may use all 109 datasets without labels or model
outcomes, but target/control features may use only the 73-table ranking subset;
causal effects are measured only on the 36-table test subset with pre-hashed
sample rosters.

The whole-row collection and representation runs have now completed.  Strict
verification found 111,616 finite 512-dimensional vectors per condition across
all 109 discovery datasets.  RoPE and No-PE use the same dataset roster, raw
inference schedule, retained-vector count, call indices and sampled row
coordinates, while every dataset has non-identical condition activations.  The
separate eight-dataset fidelity collections are also finite and coordinate
aligned; representation training selected 10,816 equal-weight validation rows
per condition from them.

Full-rank PCA-512 reconstructed both conditions with explained variance 1.0,
as expected for a non-compressing coordinate baseline.  Dense AE-384 retained
0.9941 training explained variance and passed the frozen validation gate:
0.9773 pooled, 0.9740 for No-PE and 0.9727 for RoPE.  The 8x Top-K SAE with
`k=64` retained 0.9906 on its training activations but collapsed to 0.6034 on
the disjoint fidelity roster (0.6000 No-PE and 0.6173 RoPE), so it is rejected
without validation-driven retuning. PCA passed the recipient and source live
no-op gates on the first causal table before the unequal-dose gate stopped the
run; those gates must still pass independently on every executed table. Dense
AE has not undergone a live model-causal run. These are reconstruction and
plumbing results, not evidence of a positional mechanism or downstream
improvement.

Because the unsupervised representation dictionary used all 109 discovery
datasets, the 36 causal-test datasets are disjoint from feature ranking and
model outcomes but are not representation-level untouched holdouts.  Results
from the legacy step-250000 pair therefore remain explicitly exploratory.

The first outcome-free whole-row PCA ranking over the frozen 73-table subset
also completed. It selected PCA coordinates `0`, `7`, `5`, and `3` as targets
and `181`, `502`, `153`, and `310` as their pre-frozen matched controls. The
four target scores were approximately `15.3224`, `5.3421`, `4.7173`, and
`3.4854`, with nonzero evidence on all 73 ranking tables. This establishes a
large condition-associated representation shift, not causal importance.

The first live whole-row causal attempt then stopped safely on the first table,
before publishing any target, control, or donor prediction. The decoded
target/control ablation displacement ratio was `7.81832`, exceeding the frozen
maximum of `1.25`; consequently no causal output directory or performance
result exists. This exposed an unfair intervention-dose comparison rather than
a model, checkpoint, or accelerator failure.

An outcome-free amendment is now frozen in
`analysis/pe_mechanism/protocols/whole-row-causal-dose-amendment-v1.json`.
For each raw official model call and separately for ablation and donor patches,
it shrinks only the latent edit that produced the larger decoded RMS dose,
never amplifies either side, preserves the smaller edit byte-for-byte, and
reruns the decoded-dose gate on the actual injected activations after explicitly
rejecting any per-side dose increase. The executed interventions must be
reported as dose-matched partial edits. The original split, target ranking, and
random-control draw stay
unchanged; the amendment is content-bound through ranking and causal manifests.

The amendment implementation passed 390 mechanism regressions, Ruff, package
build and the repository-wide public-history gate before direct push. A clean
checkout of exact analysis SHA `fd3b1d9addafc46533d1cb461e347f46ab2c8c96`
then regenerated the 73-table ranking. Targets, controls and all median scores
were byte-for-byte identical to the original ranking, while the new manifest
binds the amendment digest.

The single pre-declared eight-table RoPE-source to No-PE-recipient campaign
subsequently completed on a non-H100 accelerator. All recipient/source no-op,
round-trip, alignment, lineage, per-call dose and atomic-output gates passed.
Unequal full edits reached a symmetric decoded-dose ratio of 37.63; the maximum
ratio actually executed after matching was 1.000000064, with no amplification.
The maximum no-op probability difference was `6.38e-6` and maximum
reconstruction MSE was `3.78e-13`.

The causal result is a credible exploratory negative. Target ablation was not
more damaging than matched-control ablation: macro target-minus-control mean
log loss was `-7.83e-4`, median `-5.83e-4`, positive on 3/8 datasets, with
descriptive exact sign-flip `p=0.2422`. Target donor patches were effectively
indistinguishable from matched-control donor patches: mean log-loss improvement
`1.56e-4`, median `2.25e-5`, positive on 4/8 datasets, descriptive
`p=0.5547`. The reverse direction and dense AE are therefore not run merely to
search for a positive result. The sanitized finding is recorded in
`analysis/pe_mechanism/findings/whole-row-pca-dose-matched-exploratory-negative-v1.json`.

A real released TabPFN v2.6 checkpoint is not locally available to this
workstream. The mutable descendant tree now contains an offline official
fixed-weight TALENT runner for `Wp+b`, `Wp`, `b`, and zero-position conditions,
but its real-model run remains gated on official license acceptance, token-based
checkpoint access, content verification, a new clean pushed analysis commit,
and a clean TabPFN v7.1.1 checkout.

Still required before a mechanistic claim:

- preserve the completed whole-row PCA result as a credible exploratory
  negative; do not run the reverse donor direction or qualifying dense AE
  merely to search for a positive result. The failed Top-K SAE remains
  ineligible;
- acquire and content-bind the licensed released TabPFN v2.6 checkpoint before
  running its registered positional-term decomposition;
- obtain matched formal RoPE, Temporary and No-PE checkpoints before any
  validation selection or held-out TALENT/TabArena confirmation. The legacy
  pair cannot be promoted into that workflow.

No supported sparse feature, downstream improvement, or efficiency gain is
claimed at this stage. The negative mechanism result is exploratory; existing
pilot checkpoints remain diagnostic and cannot be promoted to confirmatory
three-arm evidence.

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
