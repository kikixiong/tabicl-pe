# PE Mechanism Analysis

This is an independent analysis package for studying how positional encodings
affect TabICLv2 and TabPFN v2.6. It is deliberately separate from the immutable
formal-training candidate and does not change the public `tabicl` API.

The command-line interface exposes twelve workflows:

```text
pe-mechanism collect          --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism official-collect --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism ablate           --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism localize         --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism tabpfn-localize  --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism train-repr       --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism rank-condition-shift --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism reconstruction-sensitivity --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism model-causal     --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism select-features  --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism confirm-features --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism tabarena-evaluate --config CONFIG.json --output-dir /absolute/external/run
```

`collect` is a low-level importer for already generated activation arrays.
`official-collect` is the scientific TabICLv2 route: it fits the public sklearn
classifier on training rows only, evaluates on validation or test rows, and
captures bounded activations inside the unchanged official ensemble path.
`ablate` aggregates already measured, paired ablation records; it does not run
a model. `localize` is the discovery-only fixed-weight TabICLv2 route for a
content-verified, same-step Stable RoPE/No-PE pilot pair. It evaluates the
native models alongside an explicit RoPE no-op, all-off, query-only, key-only,
phase, frequency-band, and block-local conditions while keeping official
preprocessing and inference paired. `tabpfn-localize` is the discovery-only
official TabPFN v2.6 route. It fits one fixed-seed classifier on each TALENT
training split and, without refitting, compares the native positional
projection (`Wp+b`) with weight-only (`Wp`), bias-only (`b`), and zero-position
conditions on validation rows. `reconstruction-sensitivity` is an
activation-space diagnostic and is not a causal model result.

`tabarena-evaluate` runs the complete 38-task TabArena v0.1 classification
roster at split 0 for an exact-step RoPE/No-PE pilot pair and the released
TabICL reference. All three systems use one estimator, seed 42, no added
augmentation, the same explicit classifier settings, and no automatic model
download. The workflow verifies the checkpoint pair, its checksum marker, all
source revisions, and the public roster before constructing exactly 114 jobs.
It writes a flat, atomically published private run containing per-task metrics,
paired summaries, runtime provenance, and an archive of the freshly generated
TabArena result cache. This is a same-budget exploratory comparison, not a
leaderboard reproduction or a substitute for matched formal
RoPE/Temporary/No-PE checkpoints. Start from
[`examples/tabarena-evaluate.example.json`](examples/tabarena-evaluate.example.json)
and schedule the full run with
[`scripts/slurm_tabarena_evaluate.sh`](scripts/slurm_tabarena_evaluate.sh).
Submit it from a private working directory and explicitly route both scheduler
streams outside every source checkout; otherwise Slurm's default
`slurm-<job>.out` would dirty the verified analysis tree before the program can
start. For example:

```bash
sbatch --chdir=/absolute/private/slurm-work \
  --output=/absolute/private/logs/tabarena-%j.out \
  --error=/absolute/private/logs/tabarena-%j.err \
  analysis/pe_mechanism/scripts/slurm_tabarena_evaluate.sh
```

Pass the four required `PE_*` variables with `--export` or from a private
submit wrapper. Do not submit the public script directly from its checkout.

## Exploratory BeyondArena smoke

[`scripts/run_fingerprint_beyondarena_exploratory.py`](scripts/run_fingerprint_beyondarena_exploratory.py)
adapts the same fixed one-estimator RoPE/Fingerprint/released comparison to
BeyondArena. Its default `lite` roster contains exactly one IID, one grouped,
and one temporal classification dataset. The runner resolves that explicit
roster before materialization and checks the complete 3-by-roster job matrix,
so a smoke invocation cannot silently expand into the full 142-dataset suite.
Use repeated `--dataset-name` arguments to choose another bounded,
classification-only, text-free roster.

The output is exploratory: it reports per-task errors, scale-free wins and
mean ranks by split regime, but it is neither the recommended BeyondArena
`core` protocol nor a leaderboard reproduction. Persistent Data Foundry and
materialized task caches must be routed outside Git with
`PE_BEYONDARENA_CACHE`; raw result caches are archived only inside the private
output directory. Schedule the GPU run with
[`scripts/slurm_fingerprint_beyondarena_exploratory.sh`](scripts/slurm_fingerprint_beyondarena_exploratory.sh)
and provide scheduler stdout/stderr paths externally.

`--output-dir` is always required. It must be an absolute path outside the Git
source tree. Checkpoints, activations, predictions, logs, caches, and generated
test data must stay outside this repository.

The fixed-weight localization example is
[`examples/localize-matched-step.example.json`](examples/localize-matched-step.example.json).
Replace every `/absolute/...` path, placeholder digest, and source revision;
list the complete intended TALENT discovery roster; provide a precommitted
SHA-256 for every consumed TALENT file; explicitly freeze the estimator
count, batch size, AMP, FlashAttention, and random state; and keep the output
inside the configured `private_study_root`. Object-backed arrays from a trusted
local TALENT corpus require `trusted_pickle=true`; outputs re-encode labels as
portable integer class indices and never publish object arrays. The matched pilot pair remains
`formal_eligible=false`, so this workflow cannot supply validation or held-out
three-arm evidence.

The private result records the canonical meaning of every condition, the
probability-column class order, hashes of the exact stored float32 prediction
arrays, the paired official ensemble schedule, and a path-free runtime/version
attestation. These fields make metrics independently recomputable without
loading pickle-backed output artifacts.

## Official TabPFN v2.6 fixed-weight localization

Start from the fully placeholder-only
[`examples/tabpfn-localize.example.json`](examples/tabpfn-localize.example.json).
Replace every `/absolute/...` value and every placeholder hash. The example
describes a numeric-only TALENT dataset; if a dataset has categorical arrays,
also list every consumed `C_train.npy`, `C_val.npy`, and `C_test.npy` file and
its precommitted SHA-256. The dataset manifest must assign the complete roster
to `discovery`; this command fits `train`, evaluates `val`, and refuses the
reserved test split.

The run has external prerequisites that this repository cannot satisfy:

- Read and accept the official
  [TabPFN 2.6 non-commercial license](https://huggingface.co/Prior-Labs/tabpfn_2_6/blob/main/LICENSE)
  before using the weights, and confirm that the intended use is permitted.
- Follow the official
  [gated-model access instructions](https://docs.priorlabs.ai/how-to-access-gated-models).
  For token-based acquisition from a notebook or headless client, expose the
  account API key as `TABPFN_TOKEN`; the documented manual download is an
  alternative. Never place the token in a config, log, manifest, or Git
  repository.
- Acquire the v2.6 classifier checkpoint outside this workflow, place
  `tabpfn-v2.6-classifier-v2.6_default.ckpt` locally, and record its measured
  SHA-256 in the private config. The runner disables downloads and has no
  fallback checkpoint, so a missing or mismatched file fails closed.
- Use a clean checkout of the audited TabPFN `v7.1.1` source, bind its full Git
  SHA in both training- and model-code provenance, and bind the clean analysis
  checkout separately. A package with the same version label is not sufficient
  provenance.

Once those prerequisites are satisfied, submit the real run through
[`scripts/slurm_tabpfn_localize.sh`](scripts/slurm_tabpfn_localize.sh). The
wrapper fixes one normal-partition accelerator, the short QOS and a three-hour
limit; it does not request an H100. Supply absolute `PE_ANALYSIS_ROOT`,
`PE_TABPFN_ROOT`, `PE_CONFIG`, `PE_OUTPUT_DIR`, `PE_RUNTIME_ROOT`, and
`PE_PYTHON` paths. The working directory, scheduler logs, private config,
output, and runtime cache must all stay outside both verified source checkouts.
`PE_OUTPUT_DIR` must be fresh and `PE_RUNTIME_ROOT` must already exist as a
real directory. The wrapper forces offline model access and removes
`TABPFN_TOKEN`, `HF_TOKEN`, and `HUGGING_FACE_HUB_TOKEN` before Python starts;
acquisition credentials therefore cannot enter the inference environment or
its logs.

For every dataset, the official estimator is fitted once on training rows with
one ensemble member and a fixed random state. The same fitted estimator then
produces paired `Wp+b`, `Wp`, `b`, and zero-position probabilities; the full
policy must be byte-identical to native `predict_proba`, and every temporary
hook must restore exactly. The private run publishes `predictions.npz`,
`results.json`, `summary.json`, and `manifest.json` atomically. Predictions are
stored as float32, labels as portable int64 class indices, and raw class values
are never published: only their type and content hash are recorded. This is a
component-localization experiment, not evidence that any component is a causal
mechanism or that the result generalizes beyond the discovery roster.

## Scheduled TabICL localization

For one scheduled H100 run, use
[`scripts/slurm_fixed_weight_localization.sh`](scripts/slurm_fixed_weight_localization.sh).
It accepts only absolute paths through five environment variables and sets an
isolated two-root `PYTHONPATH`: `PE_ANALYSIS_ROOT` is this analysis package
directory, while `PE_MODEL_ROOT` is the exact TabICL source checkout bound to
the pilot checkpoints. `PE_CONFIG`, `PE_OUTPUT_DIR`, and `PE_PYTHON` name the
config file, private output directory, and interpreter respectively. For
example:

```bash
PE_ANALYSIS_ROOT=/absolute/src/tabicl-pe/analysis/pe_mechanism \
PE_MODEL_ROOT=/absolute/src/pilot-tabicl \
PE_CONFIG=/absolute/config/localize-matched-step.json \
PE_OUTPUT_DIR=/absolute/private/pe-localization/run \
PE_PYTHON=/absolute/env/bin/python \
sbatch --chdir=/absolute/external/slurm-work \
  analysis/pe_mechanism/scripts/slurm_fixed_weight_localization.sh
```

Supply scheduler stdout and stderr destinations externally if needed; the
wrapper deliberately contains no machine-specific artifact or log path.

For strict activation collection, use
[`scripts/slurm_official_collect.sh`](scripts/slurm_official_collect.sh) with
the same five environment variables. The bound TabICL revision constructs the
normalization-view dictionary through a Python set, so separate condition
processes can otherwise receive different view orders despite an equal model
seed. The wrapper freezes `PYTHONHASHSEED=0`; `train-repr` additionally compares
the actual normalization views, shuffles, call indices, and sampled coordinates
and rejects any residual mismatch. Do not feed manually reordered indexes or
activation shards into representation training.

## Provenance and privacy

Every published run manifest contains only portable identifiers and content
hashes. Its schema has no path, host, scheduler, or job-ID fields, and it does
not accept arbitrary metadata. In particular, never add cluster paths, raw
predictions, checkpoint contents, scheduler identifiers, credentials, or
personal information to a manifest or Git commit.

The model adapters share a small interface for loading a model, declaring
well-defined activation sites, capturing activations with axis metadata,
applying reversible interventions, and producing predictions. Model-specific
code belongs in its adapter; benchmark preprocessing does not belong in the
shared interface.

## Evidence flow

A model-causal result has the following verified ancestry:

```text
official-collect run(s) -> train-repr run -> model-causal run
```

`train-repr` accepts only completed `collect` manifests and their declared
activation artifacts. For a dictionary shared across conditions, it separately
binds each checkpoint and requires the same data roster, official ensemble
views, feature/class shuffles, feature grouping, activation axes, and sampled
coordinates. Equal array shapes alone are not alignment evidence.

Each representation dictionary selects exactly one activation site. A strict
parent collection may contain additional captured sites, but the selected site
must be declared by that parent and every referenced activation artifact must
belong to the selected site; the unused sites are never imported implicitly.

Dense autoencoders, exact PCA, and top-k sparse autoencoders are available.
Representation fidelity is measured on datasets excluded from representation
training. Passing reconstruction is necessary but does not establish a
mechanism.

`rank-condition-shift` is the exploratory feature-freezing step after a
qualified shared representation has completed. It accepts only the configured
feature-ranking roster from the content-bound RoPE and No-PE official
collection parents already registered by `train-repr`. For each dataset it
computes the RMS aligned latent difference, scales every coordinate by its
decoder-direction norm after mapping that direction back through the
normalizer into raw activation units, divides by the pooled raw-activation RMS,
then takes the median score across datasets. It selects one target set and one
activation-frequency/raw-decoder-norm matched random-control set for both donor
directions. The config explicitly names the parent run, its manifest digest,
the frozen split-protocol digest, the exact ranking dataset roster and both
conditions' collect references. The command reads no model-causal predictions
or outcomes and publishes only a path-free
`selection.json` plus its manifest in one atomic directory transaction. This
legacy checkpoint workflow remains discovery-only and cannot establish a
formal mechanism or downstream improvement.

Ranking-bound paired whole-row runs additionally bind
`protocols/whole-row-causal-dose-amendment-v1.json` by exact digest. That
amendment was frozen after the original decoded-dose gate stopped the first
causal attempt, but before any target, control, or donor prediction was
published. It does not change the split or reselect features from model
outcomes; it makes the intervention comparison fair at execution time.

## Official model-causal runs

`model-causal` consumes one completed, strict `train-repr` run and one raw
TALENT classification dataset. The representation run directory, TALENT data
directory, checkpoint, Git roots, dataset manifest, and sample roster must be
absolute paths. The fit split is fixed to `train`; evaluation may use `val` or
`test` (default), but never `train` or `train+val`.

The parent run must declare a verified `model.pt` artifact whose source lineage
contains the requested condition and checkpoint. The runner uses the official
cache-free `TabICLClassifier` preprocessing and ensemble path. Its registered
scope is native classification with at most 10 classes and `same` feature
grouping. The model's actual identity mode must match the condition. No-op
limits cannot be relaxed beyond 0.01 reconstruction MSE, 0.02 maximum
probability deviation, or 0.005 absolute accuracy difference.

Formal paired interventions are restricted to `row_interactor`. At this site,
`target_features` and `control_features` identify coordinates in the frozen
PCA/autoencoder/sparse-autoencoder representation of the flattened CLS-slot
output of RowInteraction. They are not raw table columns, row positions,
attention heads, positional-encoding indices, or individual model neurons.

For a ranking-bound exploratory paired run, target and matched-control edits
use per-call decoded-RMS dose matching. Ablation and independently sourced
donor patches are matched separately against the recipient no-op
reconstruction. The smaller full-edit dose becomes the common dose and only
the larger latent edit is shrunk; the smaller edit is reused byte-for-byte.
The changed latent is decoded again, and the implementation rejects any
post-cast per-side dose increase before the existing maximum symmetric-ratio
gate checks the exact activations that will be injected. A zero or non-finite
dose fails closed. These are therefore
dose-matched partial edits, not complete deletions or transplants, and effects
from the ablation and donor families must not be compared as though their doses
were equal to each other.

Dataset identity is explicitly labelled `discovery`, `validation`, or
`held_out` and checked against the committed roster. Discovery and validation
runs may select candidate controls. A held-out run must instead consume a
previously hashed freeze artifact containing its numerical latent baseline,
target/control features, representation digest, and sample roster; it cannot
adapt those choices using held-out activations.

An `exploratory_pilot` checkpoint study is diagnostic-only and may be used only
with the discovery split. It records `formal_trust_verified=false` and cannot
feed validation, feature selection, or held-out confirmation. Those stages
require a fully verified formal checkpoint chain.

The runner verifies freeze content and validation-run ancestry, but a local
hash cannot prove chronology. A confirmatory held-out claim also needs the
freeze artifact committed and pushed, or recorded by an independent immutable
timestamp, before the held-out run begins.

Restoring the exact latent from the same forward pass is published only as a
round-trip plumbing control. It is not called a causal rescue. A mechanistic
rescue requires an independently bound paired activation source.

## Validation selection and held-out confirmation

`select-features` consumes only strict, completed validation `model-causal`
runs. Every candidate must use the same validation dataset roster and a formal,
self-hashed checkpoint-study attestation. Effects are recomputed from the
manifest-bound per-sample probabilities, classes, and true labels; published
mean losses or pass flags are not trusted.

Each candidate is one intersection-union hypothesis with six required positive
components: target damage, damage beyond its matched control, donor rescue,
rescue beyond a matched donor control, source native advantage, and source
no-op advantage. The candidate p-value is the maximum of those six one-sided
paired sign-flip p-values. Benjamini-Yekutieli correction is then applied once
across the candidate p-values, so the controlled discovery unit is a candidate,
not an individual endpoint. All six bootstrap confidence and cross-dataset
replication gates must also pass. At most two eligible candidates are frozen,
ranked by the weakest of their four candidate-specific mean effects; held-out
data are never used for ranking or top-k selection.

The freeze binds the validation dataset fingerprints, exact held-out sample
rosters, checkpoint/source lineage, random seed, resampling counts, and power
thresholds. Duplicate raw-data or invariant-prediction fingerprints are
rejected within validation, within held-out evaluation, and across the two
splits. This rejects exact-content aliases and aliases that preserve the bound
prediction fingerprint; it does not prove that independently serialized or
transformed datasets are semantically unrelated.

`confirm-features` requires the complete frozen candidate-by-dataset Cartesian
set and reruns the selection statistics before accepting the freeze. On held-out
data it again uses the maximum of the same six component p-values for each
candidate, then applies Holm correction across every frozen candidate. It never
reselects or keeps only the best held-out result. Sign flips are enumerated
exactly for at most 20 datasets. Larger rosters, including the complete
24-dataset TALENT held-out roster, use the frozen seed and resampling count;
their Monte Carlo p-value resolution must be fine enough for the pre-registered
Holm rank-one threshold.

The confirmation command returning exit status zero means that the complete
analysis finished and its artifacts were committed atomically. It does not mean
that a candidate confirmed; inspect `any_candidate_confirmed`,
`all_frozen_candidates_confirmed`, and each candidate's `confirmed` field.
If selection freezes no candidates, use an empty `heldout_runs` list. The
command then records `status=no_frozen_candidates` and
`heldout_evidence_consumed=false`; it does not read held-out predictions or
claim a vacuous confirmation.

The selection configuration must pre-register the canonical five-entry
`evidence_family`, candidate feature/control pairs and validation run-manifest
hashes, a sorted held-out sample-roster mapping, `maximum_selections` (one or
two), the paired source/recipient direction, bootstrap and sign-flip counts,
and the confirmation alpha, replication fraction, and random seed. The
confirmation configuration contains only the strict selection run directory
and manifest hash plus the exact frozen candidate-by-held-out-run Cartesian
list. Neither command accepts an adaptive held-out top-k option.

Start from
[`examples/select-features.example.json`](examples/select-features.example.json)
and
[`examples/confirm-features.example.json`](examples/confirm-features.example.json).
Every `/absolute/...` path and example digest is a placeholder that must be
replaced with the verified local file and its exact SHA-256. The selection
example pre-registers one candidate over the minimum eight validation and eight
held-out datasets. After selection completes, construct the confirmation
`heldout_runs` list from the frozen candidate IDs and include every frozen
candidate × held-out dataset pair exactly once; do not assume that the example
candidate was selected. If none was selected, replace the example list with an
empty list.

The sample roster is a small JSON object whose order defines the published
per-sample order:

```json
{
  "dataset_id": "dataset-id",
  "split": "test",
  "row_indices": [0, 7, 12],
  "sample_ids": ["test-0", "test-7", "test-12"]
}
```

Each `model-causal` run publishes `predictions.json`, `summary.json`, and
`manifest.json`. A `select-features` run instead publishes the selection,
freeze, summary, and manifest artifacts; a `confirm-features` run publishes the
confirmation, summary, and manifest artifacts. Their manifests bind the exact
parents, sample rosters, raw TALENT inputs, configurations, checkpoints, and
Git evidence by digest; none of their filesystem paths are serialized.

The TabPFN v2.6 adapter provides exact additive-position decomposition and
scoped instrumentation tests. Real TabPFN evidence must use the licensed,
version-pinned checkpoint and strict `tabpfn-localize` provenance described
above; synthetic adapter tests are not reported as benchmark evidence.

## Recorded exploratory findings

The matched-step legacy TabICLv2 whole-row PCA study is recorded in
[`findings/whole-row-pca-dose-matched-exploratory-negative-v1.json`](findings/whole-row-pca-dose-matched-exploratory-negative-v1.json).
After per-call target/control doses were matched without amplification, neither
target ablation nor RoPE-to-No-PE target donor patches consistently exceeded
their frozen matched controls across the eight-table causal roster. This is a
credible exploratory negative for the selected PCA directions, not evidence
that positional encoding is irrelevant and not a formal three-arm result. The
pre-declared stopping rule therefore forbids expanding to the reverse direction
or dense autoencoder merely to search for a positive result.

## Development tests

Keep the environment, bytecode, and test caches outside the source tree. One
safe local setup is:

```bash
uv venv /tmp/tabfm-pe-mechanism-test-env
uv pip install --python /tmp/tabfm-pe-mechanism-test-env/bin/python -e './[test]'
uv pip install --python /tmp/tabfm-pe-mechanism-test-env/bin/python -e './analysis/pe_mechanism[test]'
PYTHONDONTWRITEBYTECODE=1 /tmp/tabfm-pe-mechanism-test-env/bin/python -m pytest -q -p no:cacheprovider analysis/pe_mechanism/tests
```

The low-level array workflows can use the analysis package by itself. The
`official-collect` and `model-causal` workflows additionally require the exact
TabICL source checkout recorded in provenance, so install this repository's
root package in the same environment as shown above. Do not silently substitute
a separately released TabICL build for that bound source tree.
