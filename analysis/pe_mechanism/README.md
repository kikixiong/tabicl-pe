# PE Mechanism Analysis

This is an independent analysis package for studying how positional encodings
affect TabICLv2 and TabPFN v2.6. It is deliberately separate from the immutable
formal-training candidate and does not change the public `tabicl` API.

The command-line interface exposes six workflows:

```text
pe-mechanism collect          --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism official-collect --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism ablate           --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism train-repr       --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism reconstruction-sensitivity --config CONFIG.json --output-dir /absolute/external/run
pe-mechanism model-causal     --config CONFIG.json --output-dir /absolute/external/run
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

Dataset identity is explicitly labelled `discovery`, `validation`, or
`held_out` and checked against the committed roster. Discovery and validation
runs may select candidate controls. A held-out run must instead consume a
previously hashed freeze artifact containing its numerical latent baseline,
target/control features, representation digest, and sample roster; it cannot
adapt those choices using held-out activations.

The runner verifies freeze content and validation-run ancestry, but a local
hash cannot prove chronology. A confirmatory held-out claim also needs the
freeze artifact committed and pushed, or recorded by an independent immutable
timestamp, before the held-out run begins.

Restoring the exact latent from the same forward pass is published only as a
round-trip plumbing control. It is not called a causal rescue. A mechanistic
rescue requires an independently bound paired activation source.

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

The workflow publishes only `predictions.json`, `summary.json`, and
`manifest.json`. The manifest binds the exact parent manifest, representation
model, sample roster, raw TALENT arrays, configuration, checkpoint, dataset
manifest, and Git evidence by digest; none of their filesystem paths are
serialized.

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
