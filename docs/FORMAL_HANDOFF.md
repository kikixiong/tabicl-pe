# Formal Hardware Handoff

This handoff begins only after the candidate commit is final. Hardware evidence
must be generated from a clean detached checkout of that exact commit; changing
code or documentation afterward creates a different candidate.

## 1. Prepare the exact candidate

Use an isolated interpreter and explicitly disable user-site imports. Verify
the remote branch resolves to the intended commit, the checkout is detached and
clean, Git object integrity passes, and the tracked tree matches the expected
tree. Put pytest caches, bytecode, temporary directories, logs, checkpoints,
and evidence outside the checkout.

Every compute case independently repeats the commit/tree/source/import
attestation. A successful attestation from another case cannot be reused.

## 2. Run the canonical twelve-case H100 matrix

The case set is exact:

1. Three Stage 1 one-step functional cases: `rope`, `temporary`, and `none`.
2. Three Stage 2 cases at an observed fixed sequence length of 10,240.
3. Three Stage 3 cases at an observed fixed sequence length of 60,000 with
   activation recomputation enabled.
4. One Temporary CUDA RNG isolation-and-resume case.
5. One two-GPU NCCL case covering distributed state restore and the final
   cross-rank barrier.
6. One real GraphSCM prior/DataLoader tiny uninterrupted-versus-resume case with
   worker and prefetch state exercised.

The three one-step cases establish functionality only. The six Stage 2/3
max-sequence cases establish utilization: use one-second unfiltered sampling,
an explicitly signaled active training window, at least ten active samples on
every allocated GPU, and a per-GPU arithmetic mean of at least 80%. If a case
is too short, repeat iterations at the same fixed shape; do not change the
formal sequence length. All nine Trainer cases use one H100 and save a full
checkpoint bounded by the precommitted `C`. The NCCL case is the only two-GPU
case.

Assemble the final smoke attestation only after all twelve case directories,
runtime evidence, completion receipts, checkpoint hashes/sizes, and (for the
six utilization cases) raw-window summaries validate as one exact set.

## 3. Build and validate the fresh formal overlay

The overlay fixes three arms by three stages, all as held one-H100 jobs with
budgets 500,000/40,000/10,000. It binds the exact candidate, environment,
protocol digests, prior, architecture, optimizer, scientific configuration,
the twelve-case smoke attestation, `C`, and durable-output ceilings.

Before the first scheduler submission, the controller must validate the smoke
attestation and enforce the byte-exact capacity rule documented in
`FORMAL_STATUS.md`. Insufficient space is a hard block, not a warning.

Submission is one transaction: submit all nine jobs held, atomically publish
the protocol ledger and held-job receipt, then release them. Stage 2 and Stage
3 use same-arm `afterok` dependencies with invalid-dependency termination. If
any of the nine submissions or any release fails, cancel only the jobs accepted
by that transaction in reverse order. If cancellation is incomplete, publish a
truthful recovery record.

## 4. Monitor without mutating training

`scripts/monitor_formal_identity.py` is the standalone read-only monitor. Call
it with absolute manifest, state-directory, and event-ledger-directory paths.
It defaults to a bounded 30-minute run with 30-second polling, supports
`--once`, and uses a nonblocking per-state-directory lock. Healthy polls are
silent; only anomalies, scheduler/stage transitions, and checkpoint issues are
emitted and appended to the bounded event ledger.

The monitor warns below 22 GiB available and emits a submit-block condition
below 20 GiB. It queries only scheduler/accounting status, reads bounded log
tails, and rechecks checkpoint ZIP/hash integrity only when checkpoint path
identity or modification metadata changes. It never invokes a submit, release,
or cancellation command.

## 5. Stop conditions

Do not start the nine formal jobs if any exact-source/import check, one of the
twelve hardware cases, checkpoint ceiling, nonfinite/OOM scan, runtime
attestation, capacity gate, or atomic-publication test fails. Preserve the
failed evidence outside the public repository, fix the candidate, and rerun the
entire hardware matrix because a code change creates a new exact commit.

Evaluation code and datasets remain a separate follow-on workstream and must
not be added to this training candidate.
