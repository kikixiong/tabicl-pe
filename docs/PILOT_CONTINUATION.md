# Legacy RoPE/No-PE pilot continuation

This runbook covers only the exploratory seed-42 pilot continuation requested
after the August 2026 checkpoint audit. It does not create the formal
three-arm cohort and does not make the legacy checkpoints paper-usable.

## Exact continuation edges

The targeted submitter creates exactly these jobs:

1. Stable RoPE Stage 1: validated step 479000 to terminal step 500000.
2. Stable RoPE Stage 2: fresh child loaded from the validated RoPE 500000
   parent, with an `afterok` dependency on the Stage 1 wrapper.
3. No-PE Stage 2: full-state resume from validated step 6200 to terminal step
   40000; it has no dependency because No-PE Stage 1 already reached 500000.

It never calls `submit_h100_rope_none_full.sh`, never resubmits No-PE Stage 1,
and never creates Stage 3 or Temporary-arm jobs.

Audited starting checkpoint identities:

| Checkpoint | SHA-256 |
|---|---|
| RoPE Stage 1 step 479000 | `7f36e035e586ddbd2d33fa0e0cb7f649456946569dbaeb9335a4e7907654e117` |
| No-PE Stage 1 step 500000 | `14efa93349cb7b6f7d458508012a084526d970aeb00749bbb1cebbbe38aaec87` |
| No-PE Stage 2 step 6200 | `34564fe378d57d89b10e800b01ab63185170df25ad9210175f162c31749a16be` |

The wrapper verifies ZIP CRC, safe loading, mode, step, finite model and
optimizer tensors, scheduler position, file stability, and SHA where it is
known. It also locks the historical trainer and Stage 1/2 script byte digests.

## Submission procedure

Dry-run is the default and creates no jobs or artifacts:

```bash
scripts/submit_h100_pilot_continuation.sh --dry-run
```

Before an actual submission, commit the wrapper and tests, ensure the checkout
is clean, rerun the dry-run, inspect `squeue -p h100` and `df -h`, and then use:

```bash
scripts/submit_h100_pilot_continuation.sh --submit
```

The actual path records the committed source SHA in a fresh receipt. If a
multi-job submission fails, it cancels only job IDs created by that invocation.
It never touches pre-existing jobs. No submission has been made merely by
adding or testing these scripts.

## Checkpoint snapshots

Before No-PE Stage 2 starts, and after RoPE Stage 1 reaches 500000, the wrapper
creates or verifies an immutable copied checkpoint and a self-hashed canonical
manifest under:

```text
artifacts/pilot-continuation-snapshots/<continuation-id>/<mode>/stage1-step500000/
```

The snapshot manifest explicitly labels the artifact exploratory and records
that its source identity is based on operational history rather than embedded
checkpoint provenance.

## Stage 2 utilization semantics

Stage 1 retains the historical hard gate requiring an active-window H100 mean
of at least 80%. Stage 2 still calculates and reports the same statistic, but
low utilization is diagnostic-only for this legacy pilot. This prevents a
validated step-40000 checkpoint from being marked failed solely because the
variable-length Stage 2 workload averaged below 80%.

OOM, training-process failure, missing or corrupt checkpoints, non-finite
checkpoint tensors, wrong mode/step, insufficient disk reserve, missing FA3,
or source/checkpoint mismatch remain hard failures. This diagnostic exception
must not be copied into the formal protocol, whose locked utilization gate is
unchanged.

## Read-only 30-minute monitor

After the submitter writes its receipt, a one-time read-only check is:

```bash
.venv/bin/python scripts/monitor_pilot_continuation.py \
  --receipt artifacts/pilot-continuation/<continuation-id>/submission-receipt.json \
  --once
```

For a persistent 30-minute monitor on the CPU partition:

```bash
sbatch \
  --output="artifacts/logs/pilot-monitor-%j.out" \
  --error="artifacts/logs/pilot-monitor-%j.err" \
  --export="ALL,PILOT_ROOT=$PWD,PILOT_RECEIPT=$PWD/artifacts/pilot-continuation/<continuation-id>/submission-receipt.json" \
  scripts/slurm_monitor_pilot_continuation.sh
```

The monitor only queries Slurm and reads disk, logs, checkpoints, GPU CSVs, and
snapshot manifests. It checks disk warnings at 22 GiB and critical reserve at
20 GiB, checkpoint CRC and age, step progress, common fatal log signatures,
per-GPU means, stage transitions, and terminal states. It never cancels,
requeues, submits, writes, or prunes training artifacts.

## Scientific limitation

Legacy checkpoints contain model, optimizer, scheduler, step, and model config
only. They do not contain a source SHA, exact RNG/DataLoader trajectory, AMP
scaler state, or parent lineage. The trainer deliberately reseeds from the
restored step, but that is not proof of uninterrupted-versus-resumed equality.
Consequently all outputs from this continuation remain exploratory pilots.
