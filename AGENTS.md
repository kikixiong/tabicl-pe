# Repository Guidance

These instructions apply to the whole repository.

The formal position-versus-identity study is a matched three-arm pre-training
experiment: `rope`, `temporary`, and `none`. Keep the canonical stage budgets
at 500,000, 40,000, and 10,000 steps. The only treatment difference permitted
within a stage cohort is `row_identity_mode` and its derived treatment digest.
The supported experiment seeds are exactly 42, 43, and 44. Each seed owns a
new fresh namespace, while all three seeds use the same immutable training
commit and frozen protocol.

Formal execution is fresh-only and must use a clean detached checkout of the
exact candidate commit advertised by `https://github.com/kikixiong/tabicl-pe.git`
at `refs/heads/codex/position-identity-v1`. Do not weaken the bounded,
digest-bound exact-ref query, source/import attestation, checkpoint and
parent validation, byte-exact capacity checks, atomic publication, retention,
or the held nine-job submission transaction. Submit the three Stage 1 jobs
first, followed by the three Stage 2 and three Stage 3 jobs with same-arm
`afterok` dependencies. Every production job is one H100 on the `long` QoS
with 64 CPUs and 128 GiB of memory.

Node-local checkout work roots must be physical directories on a different
filesystem from durable artifact namespaces. Controllers check this before
creating a namespace and again while jobs are held; compute wrappers repeat it
before `mktemp` or `git clone`.

Do not call a production cohort complete from in-job receipts alone. Require
the independent receipt-bound accounting check and write-once terminal
scheduler-log attestation after all nine allocations are successfully terminal.

Evaluation and benchmark implementation are intentionally outside the formal
training candidate. Do not add evaluation code to a training-protocol change.

Never commit cluster-specific paths, scheduler job identifiers, logs,
checkpoints, raw predictions, telemetry exports, credentials, or personal
information. Keep generated caches and test artifacts outside the source tree.

Read `docs/FORMAL_STATUS.md` before changing the formal protocol and
`docs/FORMAL_HANDOFF.md` before attempting hardware validation or submission.
