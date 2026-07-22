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
