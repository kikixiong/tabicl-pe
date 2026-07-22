# Formal Position–Identity Study Status

## Scientific question

The study asks whether TabICLv2 needs stable feature position or only temporary
feature identity. It trains three otherwise matched arms from scratch:

| Arm | Row interaction treatment | Intended contrast |
| --- | --- | --- |
| `rope` | Standard RoPE tied to input feature order | Stable ordinal position |
| `temporary` | RoPE identities randomly reassigned per table/forward pass | Identity without stable order |
| `none` | Row RoPE disabled | No explicit positional identity |

The shared stage budgets are fixed at 500,000, 40,000, and 10,000 steps. Shared
cohort protocol digests bind source, environment, prior, architecture,
optimizer/scheduler/scaler configuration, seeds, and stage settings while
excluding only the identity treatment. Arm protocol digests bind that treatment
separately. This makes the matched comparison machine-checkable.

## Current readiness

The candidate is CPU-ready. Unit, integration, resume, Gloo, shell transaction,
monitor, capacity, provenance, and public-hygiene tests are implemented. No H100
result is claimed yet, and no formal production training should be submitted
until every hardware gate in `FORMAL_HANDOFF.md` passes for the exact candidate
commit.

The formal execution path is fresh-only. Every compute case must start from a
clean detached exact-commit checkout, attest the Git commit/tree and tracked
source bytes, isolate imports from user/editable installations, and prove that
loaded `tabicl` modules come from that checkout.

## Storage and retention contract

Let `C` be the externally chosen, precommitted maximum bytes for one full
checkpoint, and let `L` be the aggregate allowance for durable logs,
attestations, manifests, and protocol metadata. A fresh study reserves

```text
R = 15*C + L
required_free_bytes = 20 GiB + ceil(5*R/4)
```

All arithmetic is integer byte arithmetic. The fifteen checkpoint slots cover
nine terminal checkpoints plus six concurrent retained/atomic peak slots. Each
active stage directory may contain at most two checkpoint-shaped files. After a
terminal checkpoint passes strict final validation, pruning must leave only
that terminal checkpoint. Later capacity checks audit the full study tree,
subtract actual consumed bytes from the same study-wide budget, and fail if the
tree changes around the filesystem snapshot.

`C` is not inferred from a convenient one-step statistic. The H100 smoke matrix
must save all nine full Trainer checkpoints, prove each is no larger than the
same precommitted `C`, and bind both `C` and the maximum observed size into its
final attestation. The formal overlay must use exactly that `C`.

## Safety boundary

Final manifests and controller ledgers use durable write-once publication, and
readers reject symlinks, malformed or noncanonical JSON, source drift, and
checkpoint mismatch. This protects against accidental drift and ordinary
partial writes. A coordinated same-owner process that replaces trusted files
after validation is outside this threat model; defending against that requires
a different security principal or cryptographic signing service.

The long-running production jobs do not produce a high-frequency GPU CSV and
make no utilization claim. A separate read-only, event-only monitor observes
scheduler state, bounded log tails, checkpoint identity/integrity, validation
reports, disk thresholds, and any already recorded utilization evidence. It
never submits or cancels jobs.
