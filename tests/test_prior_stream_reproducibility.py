import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tabicl.prior._dataset import PriorDataset


@pytest.mark.parametrize("fixed_length", [1024, 10240, 60000])
def test_equal_sequence_bounds_generate_the_exact_fixed_length(fixed_length):
    dataset = PriorDataset(
        prior_type="dummy",
        batch_size=1,
        min_features=2,
        max_features=2,
        max_classes=2,
        min_seq_len=fixed_length,
        max_seq_len=fixed_length,
        min_train_size=0.5,
        max_train_size=0.5,
        device="cpu",
    )

    X, y, _, seq_lens, train_sizes = dataset.get_batch()

    assert X.shape[1] == fixed_length
    assert y.shape[1] == fixed_length
    assert seq_lens.tolist() == [fixed_length]
    assert train_sizes.tolist() == [fixed_length // 2]


def _dataset(seed: int, cursor: int = 0) -> PriorDataset:
    dataset = PriorDataset(
        prior_type="dummy",
        batch_size=3,
        min_features=4,
        max_features=4,
        max_classes=4,
        min_seq_len=8,
        max_seq_len=9,
        min_train_size=2,
        max_train_size=6,
        device="cpu",
    )
    dataset.configure_logical_stream(
        schema="dummy-prior-v1",
        experiment_seed=seed,
        ddp_rank=0,
        world_size=1,
        cursor=cursor,
    )
    return dataset


def _batches(seed: int, *, workers: int, prefetch: int, cursor: int = 0, count: int = 5):
    from tabicl.prior._genload import make_prior_dataloader

    loader = make_prior_dataloader(
        _dataset(seed, cursor),
        num_workers=workers,
        prefetch_factor=prefetch,
        pin_memory=False,
    )
    iterator = iter(loader)
    return [next(iterator) for _ in range(count)]


def _assert_batch_equal(actual, expected):
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def _graph_scm_dataset(cursor: int = 0) -> PriorDataset:
    from tabicl.prior.graph_lib._config import PriorConfig

    dataset = PriorDataset(
        regression=False,
        prior_type="graph_scm",
        batch_size=1,
        batch_size_per_gp=1,
        min_features=2,
        max_features=2,
        max_classes=2,
        min_seq_len=64,
        max_seq_len=64,
        min_train_size=28,
        max_train_size=36,
        n_jobs=1,
        device="cpu",
        config=PriorConfig(
            min_n_nodes=2,
            max_n_nodes=4,
            fct_types="lin",
            multi_fct_types="concat",
            filter_unpredictable_graphs=True,
            filter_unpredictable_datasets=True,
        ),
    )
    dataset.configure_logical_stream(
        schema="graph-scm-hash-seed-v1",
        experiment_seed=2025,
        ddp_rank=0,
        world_size=1,
        cursor=cursor,
    )
    return dataset


def _graph_scm_batch_digests(*, workers: int, cursor: int, count: int):
    from tabicl.prior._genload import make_prior_dataloader

    loader = make_prior_dataloader(
        _graph_scm_dataset(cursor),
        num_workers=workers,
        prefetch_factor=3,
        pin_memory=False,
    )
    iterator = iter(loader)
    digests = []
    for _ in range(count):
        digest = hashlib.sha256()
        for tensor in next(iterator):
            tensor = tensor.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        digests.append(digest.hexdigest())
    return digests


def _run_graph_scm_digest_probe(hash_seed: int):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(hash_seed)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--graph-scm-digest-probe"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(result.stdout.splitlines()[-1])


def test_logical_batch_is_identical_across_workers_and_prefetch_timing():
    reference = _batches(2025, workers=0, prefetch=2)
    one_worker = _batches(2025, workers=1, prefetch=1)
    prefetched = _batches(2025, workers=2, prefetch=4)

    for expected, actual_one, actual_prefetched in zip(reference, one_worker, prefetched):
        _assert_batch_equal(actual_one, expected)
        _assert_batch_equal(actual_prefetched, expected)


def test_resume_from_cursor_matches_uninterrupted_stream_and_seed_changes_stream():
    uninterrupted = _batches(2025, workers=2, prefetch=3, count=6)
    resumed = _batches(2025, workers=1, prefetch=1, cursor=4, count=2)
    for actual, expected in zip(resumed, uninterrupted[4:]):
        _assert_batch_equal(actual, expected)

    different_seed = _batches(2026, workers=0, prefetch=2, cursor=4, count=1)[0]
    assert not torch.equal(different_seed[0], uninterrupted[4][0])


def test_dataloader_generator_is_reconstructed_instead_of_checkpointed():
    from tabicl.prior._genload import make_prior_dataloader

    source = _dataset(2025)
    source_loader = make_prior_dataloader(
        source,
        num_workers=0,
        pin_memory=False,
    )
    initial_loader_rng = source_loader.generator.get_state().clone()
    next(iter(source_loader))
    assert not torch.equal(source_loader.generator.get_state(), initial_loader_rng)

    stream_state = source.logical_stream_state_dict(cursor=1)
    assert all("generator" not in key for key in stream_state)

    resumed = _dataset(2025)
    resumed.load_logical_stream_state_dict(stream_state)
    resumed_loader = make_prior_dataloader(
        resumed,
        num_workers=0,
        pin_memory=False,
    )
    torch.testing.assert_close(
        resumed_loader.generator.get_state(), initial_loader_rng, rtol=0, atol=0
    )
    _assert_batch_equal(next(iter(resumed_loader)), source._get_logical_batch(1))


def test_real_graph_scm_is_hash_seed_worker_and_resume_invariant():
    probes = [_run_graph_scm_digest_probe(seed) for seed in (1, 2)]

    for probe in probes:
        assert probe["two_workers"] == probe["uninterrupted"]
        assert probe["resumed"] == probe["uninterrupted"][2:]
    assert probes[1:] == probes[:-1]


def test_cpu_logical_batch_never_touches_cuda_rng(monkeypatch):
    dataset = _dataset(2025)

    def cuda_forbidden(*args, **kwargs):
        raise AssertionError("CPU prior touched CUDA RNG")

    monkeypatch.setattr(torch.cuda, "is_available", cuda_forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", cuda_forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", cuda_forbidden)
    monkeypatch.setattr(torch, "manual_seed", cuda_forbidden)

    batch = dataset._get_logical_batch(3)
    assert batch[0].device.type == "cpu"


if __name__ == "__main__" and "--graph-scm-digest-probe" in sys.argv:
    print(
        json.dumps(
            {
                "uninterrupted": _graph_scm_batch_digests(
                    workers=0, cursor=0, count=3
                ),
                "two_workers": _graph_scm_batch_digests(
                    workers=2, cursor=0, count=3
                ),
                "resumed": _graph_scm_batch_digests(
                    workers=1, cursor=2, count=1
                ),
            },
            sort_keys=True,
        )
    )
