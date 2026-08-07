from __future__ import annotations

import numpy as np
import pytest

from pe_mechanism.statistics import (
    adjust_fdr,
    adjust_fdr_arbitrary_dependence,
    adjust_holm,
    paired_bootstrap_ci,
    paired_differences,
    paired_sign_flip_p_value,
    select_causal_sites,
    summarize_paired,
)


def test_paired_direction_and_summary_are_deterministic() -> None:
    baseline = [0.5, 0.6, 0.7]
    intervention = [0.6, 0.8, 0.8]
    differences = paired_differences(baseline, intervention)
    np.testing.assert_allclose(differences, [0.1, 0.2, 0.1])
    lower_a, upper_a = paired_bootstrap_ci(differences, n_resamples=500, seed=3)
    lower_b, upper_b = paired_bootstrap_ci(differences, n_resamples=500, seed=3)
    assert (lower_a, upper_a) == (lower_b, upper_b)
    summary = summarize_paired(
        baseline, intervention, n_resamples=500, seed=3
    )
    assert summary.count == 3
    assert summary.confidence_low > 0.0
    assert summary.positive_fraction == 1.0


def test_pairing_rejects_cell_level_arrays_and_missing_values() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_differences(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="complete and finite"):
        paired_differences([0.5, np.nan], [0.6, 0.7])
    with pytest.raises(ValueError, match="complete and finite"):
        paired_bootstrap_ci([0.1, np.inf])


def test_lower_is_better_flips_effect_direction() -> None:
    result = paired_differences([0.5, 0.6], [0.4, 0.3], higher_is_better=False)
    np.testing.assert_allclose(result, [0.1, 0.3])


def test_fdr_adjustment_preserves_input_order() -> None:
    adjusted = adjust_fdr([0.04, 0.001, 0.03, 0.2])
    np.testing.assert_allclose(adjusted, [0.0533333333, 0.004, 0.0533333333, 0.2])
    with pytest.raises(ValueError):
        adjust_fdr([1.1])


def test_by_adjustment_adds_harmonic_arbitrary_dependence_correction() -> None:
    p_values = [0.04, 0.001, 0.03, 0.2]
    harmonic = 1.0 + 1 / 2 + 1 / 3 + 1 / 4
    np.testing.assert_allclose(
        adjust_fdr_arbitrary_dependence(p_values),
        np.minimum(adjust_fdr(p_values) * harmonic, 1.0),
    )
    with pytest.raises(ValueError):
        adjust_fdr_arbitrary_dependence([])


def test_holm_adjustment_preserves_order_and_is_step_down_monotone() -> None:
    adjusted = adjust_holm([0.04, 0.001, 0.03, 0.2])
    np.testing.assert_allclose(adjusted, [0.09, 0.004, 0.09, 0.2])
    with pytest.raises(ValueError):
        adjust_holm([float("nan")])


def test_sign_flip_p_value_is_deterministic_and_directional() -> None:
    positive_a = paired_sign_flip_p_value([1.0] * 12, n_resamples=1000, seed=9)
    positive_b = paired_sign_flip_p_value([1.0] * 12, n_resamples=1000, seed=9)
    negative = paired_sign_flip_p_value([-1.0] * 12, n_resamples=1000, seed=9)
    assert positive_a == positive_b
    assert positive_a < 0.01
    assert negative > 0.99


def test_sign_flip_uses_exact_small_sample_null_distribution() -> None:
    assert paired_sign_flip_p_value([1.0] * 3, n_resamples=1, seed=1) == 1 / 8
    assert paired_sign_flip_p_value([1.0] * 8, n_resamples=1, seed=999) == 1 / 256


def test_site_selection_requires_replication_and_control_excess() -> None:
    selected = select_causal_sites(
        {
            "strong": [0.5] * 12,
            "random_like": [0.05] * 12,
            "inconsistent": [0.5, -0.5] * 6,
        },
        {
            "strong": [0.05] * 12,
            "random_like": [0.05] * 12,
            "inconsistent": [0.6, 0.1] * 6,
        },
        n_resamples=500,
    )
    assert [item["site"] for item in selected] == ["strong"]
    assert selected[0]["adjusted_p_value"] <= 0.05
    with pytest.raises(ValueError, match="max_sites"):
        select_causal_sites({"a": [1.0]}, {"a": [0.0]}, max_sites=-1)
