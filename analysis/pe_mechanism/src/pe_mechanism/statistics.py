"""Small paired-statistics helpers used by mechanism experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


def paired_differences(
    baseline: Sequence[float] | np.ndarray,
    intervention: Sequence[float] | np.ndarray,
    *,
    higher_is_better: bool = True,
) -> np.ndarray:
    baseline_array = np.asarray(baseline, dtype=np.float64)
    intervention_array = np.asarray(intervention, dtype=np.float64)
    if baseline_array.ndim != 1 or intervention_array.ndim != 1:
        raise ValueError("paired observations must be one-dimensional dataset-level vectors")
    if baseline_array.shape != intervention_array.shape:
        raise ValueError("baseline and intervention must have identical shapes")
    if baseline_array.size == 0:
        raise ValueError("paired observations must be non-empty")
    if not np.isfinite(baseline_array).all() or not np.isfinite(intervention_array).all():
        raise ValueError("paired observations must be complete and finite")
    raw = intervention_array - baseline_array
    return raw if higher_is_better else -raw


def paired_bootstrap_ci(
    differences: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all():
        raise ValueError("differences must be complete and finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def adjust_fdr(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values in input order."""
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and between zero and one")
    order = np.argsort(values)
    ranked = values[order]
    scale = values.size / np.arange(1, values.size + 1)
    adjusted_ranked = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def paired_sign_flip_p_value(
    differences: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float:
    """One-sided paired randomization p-value for a positive mean effect."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a non-empty finite one-dimensional vector")
    if isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    # Batch the sign flips so large dataset rosters do not allocate an
    # n_resamples-by-n_datasets array all at once.
    remaining = int(n_resamples)
    while remaining:
        batch = min(remaining, 2048)
        signs = rng.integers(0, 2, size=(batch, values.size), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        exceedances += int(np.count_nonzero((signs * values).mean(axis=1) >= observed))
        remaining -= batch
    return float((exceedances + 1) / (int(n_resamples) + 1))


@dataclass(frozen=True)
class PairedSummary:
    count: int
    mean_effect: float
    median_effect: float
    confidence_low: float
    confidence_high: float
    positive_fraction: float


def summarize_paired(
    baseline: Sequence[float] | np.ndarray,
    intervention: Sequence[float] | np.ndarray,
    *,
    higher_is_better: bool = True,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> PairedSummary:
    differences = paired_differences(
        baseline, intervention, higher_is_better=higher_is_better
    )
    low, high = paired_bootstrap_ci(
        differences, n_resamples=n_resamples, seed=seed
    )
    return PairedSummary(
        count=int(differences.size),
        mean_effect=float(differences.mean()),
        median_effect=float(np.median(differences)),
        confidence_low=low,
        confidence_high=high,
        positive_fraction=float(np.mean(differences > 0.0)),
    )


def select_causal_sites(
    site_effects: Mapping[str, Sequence[float]],
    matched_control_effects: Mapping[str, Sequence[float]],
    *,
    min_replication_fraction: float = 0.60,
    max_sites: int = 2,
    n_resamples: int = 10_000,
    seed: int = 42,
    fdr_alpha: float = 0.05,
) -> list[dict[str, object]]:
    """Select sites whose paired excess passes replication, CI, and FDR gates."""
    if site_effects.keys() != matched_control_effects.keys():
        raise ValueError("site and control mappings must have identical keys")
    if isinstance(max_sites, bool) or not isinstance(max_sites, int) or max_sites < 1:
        raise ValueError("max_sites must be a positive integer")
    if not 0.0 <= min_replication_fraction <= 1.0:
        raise ValueError("min_replication_fraction must lie between zero and one")
    if not 0.0 < fdr_alpha <= 1.0:
        raise ValueError("fdr_alpha must lie in (0, 1]")
    candidates: list[dict[str, object]] = []
    for offset, site in enumerate(sorted(site_effects)):
        observed = np.asarray(site_effects[site], dtype=np.float64)
        control = np.asarray(matched_control_effects[site], dtype=np.float64)
        if observed.shape != control.shape or observed.ndim != 1:
            raise ValueError(f"site {site!r} effects must be aligned one-dimensional arrays")
        if observed.size == 0 or not np.isfinite(observed).all() or not np.isfinite(control).all():
            raise ValueError(f"site {site!r} effects must be non-empty and finite")
        contrast = np.abs(observed) - np.abs(control)
        low, high = paired_bootstrap_ci(
            contrast, n_resamples=n_resamples, seed=seed + offset
        )
        replication = float(np.mean(contrast > 0.0))
        candidates.append(
            {
                "site": site,
                "median_absolute_excess": float(np.median(contrast)),
                "mean_absolute_excess": float(np.mean(contrast)),
                "confidence_low": low,
                "confidence_high": high,
                "replication_fraction": replication,
                "p_value": paired_sign_flip_p_value(
                    contrast, n_resamples=n_resamples, seed=seed + 100_000 + offset
                ),
            }
        )
    if not candidates:
        return []
    adjusted = adjust_fdr(float(item["p_value"]) for item in candidates)
    for item, adjusted_p in zip(candidates, adjusted, strict=True):
        item["adjusted_p_value"] = float(adjusted_p)
    candidates = [
        item
        for item in candidates
        if float(item["confidence_low"]) > 0.0
        and float(item["replication_fraction"]) >= min_replication_fraction
        and float(item["adjusted_p_value"]) <= fdr_alpha
    ]
    candidates.sort(
        key=lambda item: (float(item["median_absolute_excess"]), str(item["site"])),
        reverse=True,
    )
    return candidates[:max_sites]


def summary_dict(summary: PairedSummary) -> dict[str, object]:
    return asdict(summary)
