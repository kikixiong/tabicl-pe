# PE Mechanism Analysis

This is an independent analysis package for studying how positional encodings
affect TabICLv2 and TabPFN v2.6. It is deliberately separate from the immutable
formal-training candidate and does not change the public `tabicl` API.

The command-line interface exposes eight workflows:

```text
pe-mechanism collect          --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism official-collect --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism ablate           --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism train-repr       --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism reconstruction-sensitivity --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism model-causal     --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism select-features  --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism confirm-features --config CONFIG.json --output-dir /absolute/external/run
```

`collect` is a low-level importer for already generated activation arrays.
`official-collect` is the scientific TabICLv2 route: it fits the public sklearn
classifier on training rows only, evaluates on validation or test rows, and
captures bounded activations inside the unchanged official ensemble path.
`ablate` aggregates already measured, paired ablation records; it does not run
a model. `reconstruction-sensitivity` is an activation-space diagnostic and is
not a causal model result.

`--output-dir` is always required. It must be an absolute path outside the Git
source tree. Checkpoints, activations, predictions, logs, caches, and generated
test data must stay outside this repository.

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

Dense autoencoders, exact PCA, and top-k sparse autoencoders are available.
Representation fidelity is measured on datasets excluded from representation
training. Passing reconstruction is necessary but does not establish a
mechanism.

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
scoped instrumentation tests. Real TabPFN evidence still requires a separately
available, version-pinned released checkpoint; synthetic adapter tests are not
reported as benchmark evidence.

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
