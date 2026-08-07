# Repository Guidance

These instructions apply to the whole repository.

The formal position-versus-identity study is a matched three-arm pre-training
experiment: `rope`, `temporary`, and `none`. Keep the canonical stage budgets
at 500,000, 40,000, and 10,000 steps. The only treatment difference permitted
within a stage cohort is `row_identity_mode` and its derived treatment digest.

Formal execution is fresh-only and must use a clean detached checkout of the
exact candidate commit. Do not weaken source/import attestation, checkpoint and
parent validation, byte-exact capacity checks, atomic publication, retention,
or the held nine-job submission transaction.

Evaluation and benchmark implementation are intentionally outside the formal
training candidate. Do not add evaluation code to a training-protocol change.
Mechanism work lives only on its descendant analysis branch under
`analysis/pe_mechanism`; it must not be copied into an active training
checkout. Read `docs/PE_MECHANISM_STATUS.md` and
`docs/PE_MECHANISM_HANDOFF.md` before changing that workstream.

Never commit cluster-specific paths, scheduler job identifiers, logs,
checkpoints, raw predictions, telemetry exports, credentials, or personal
information. Keep generated caches and test artifacts outside the source tree.

Read `docs/FORMAL_STATUS.md` before changing the formal protocol and
`docs/FORMAL_HANDOFF.md` before attempting hardware validation or submission.
