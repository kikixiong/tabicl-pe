import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tabicl.train._identity_rng import (
    TrainerIdentityRNG,
    build_checkpoint_bundle,
    make_identity_treatment,
    restore_sampler_from_bundle,
    validate_identity_treatment,
)


def _gloo_identity_worker(rank, world_size, init_file):
    from tabicl.train._identity_rng import TrainerIdentityRNG
    from tabicl.train._rng_state import (
        gather_all_rank_rng_state,
        restore_all_rank_rng_state,
    )

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        sampler = TrainerIdentityRNG(
            identity_mode="temporary",
            base_seed=44,
            rank=rank,
            world_size=world_size,
        )
        sampler.sample_for_micro_batch(
            batch_size=2, num_identity_tokens=7, device="cpu"
        )
        checkpoint = sampler.checkpoint_fields()
        expected = sampler.sample_for_micro_batch(
            batch_size=3, num_identity_tokens=5, device="cpu"
        )

        restored = TrainerIdentityRNG(
            identity_mode="temporary",
            base_seed=44,
            rank=rank,
            world_size=world_size,
        )
        restored.restore_checkpoint(checkpoint, only_load_model=False)
        actual = restored.sample_for_micro_batch(
            batch_size=3, num_identity_tokens=5, device="cpu"
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert set(checkpoint["identity_sampler"]["rank_states"]) == {"0", "1"}

        random.seed(100 + rank)
        np.random.seed(200 + rank)
        torch.manual_seed(300 + rank)
        rng_bundle = gather_all_rank_rng_state(rank=rank, world_size=world_size)
        expected_rng = (random.random(), np.random.random(), torch.rand(3))
        random.random(), np.random.random(), torch.rand(3)
        restore_all_rank_rng_state(rng_bundle, rank=rank, world_size=world_size)
        actual_rng = (random.random(), np.random.random(), torch.rand(3))
        assert actual_rng[0] == expected_rng[0]
        assert actual_rng[1] == expected_rng[1]
        torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)
        assert set(rng_bundle["rank_states"]) == {"0", "1"}
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_identity_bundle_round_trips(tmp_path):
    mp.spawn(
        _gloo_identity_worker,
        args=(2, str(tmp_path / "gloo-init")),
        nprocs=2,
        join=True,
    )


def test_identity_sampling_does_not_advance_global_torch_rng():
    torch.manual_seed(1234)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    sampler = TrainerIdentityRNG(mode="temporary", seed=91, rank=0, world_size=1)
    permutations = sampler.sample_permutations(batch_size=4, num_features=13)

    assert permutations.shape == (4, 13)
    torch.testing.assert_close(torch.get_rng_state(), cpu_before)
    if cuda_before is not None:
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_before):
            torch.testing.assert_close(actual, expected)


def test_identity_sampler_state_round_trips_and_is_hash_protected():
    sampler = TrainerIdentityRNG(mode="temporary", seed=17, rank=0, world_size=1)
    sampler.sample_permutations(batch_size=2, num_features=5)
    state = sampler.state_dict()
    expected = sampler.sample_permutations(batch_size=3, num_features=7)

    restored = TrainerIdentityRNG(mode="temporary", seed=17, rank=0, world_size=1)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.sample_permutations(3, 7), expected)

    state["generator_state"][0] ^= 1
    with pytest.raises(ValueError, match="hash"):
        restored.load_state_dict(state)


def test_identity_bundle_swapped_outer_rank_keys_fail_closed():
    states = [
        TrainerIdentityRNG(
            mode="temporary", seed=17, rank=rank, world_size=2
        ).state_dict()
        for rank in range(2)
    ]
    bundle = build_checkpoint_bundle(states)
    bundle["rank_states"] = {
        "0": bundle["rank_states"]["1"],
        "1": bundle["rank_states"]["0"],
    }
    restored = TrainerIdentityRNG(
        mode="temporary", seed=17, rank=0, world_size=2
    )

    with pytest.raises(ValueError, match="rank"):
        restore_sampler_from_bundle(restored.sampler, bundle)


def test_treatment_validation_rejects_mode_seed_and_world_size_drift():
    treatment = make_identity_treatment(mode="temporary", seed=123, world_size=2)
    validate_identity_treatment(treatment, mode="temporary", seed=123, world_size=2)

    for field, value in (("mode", "rope"), ("seed", 124), ("world_size", 1)):
        kwargs = {"mode": "temporary", "seed": 123, "world_size": 2}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            validate_identity_treatment(treatment, **kwargs)
