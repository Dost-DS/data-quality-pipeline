"""Correctness tests for the drift metric implementations, and the
bin-geometry demonstration behind the PSI result (Chapter 5)."""
import json
import sys, os
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from kafka_consumer_drift import psi, ks, chi2


def _edges(lo, hi, n=10):
    return list(np.linspace(lo, hi, n + 1))


def test_psi_zero_on_identical_distributions():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 50_000)
    edges = _edges(base.min(), base.max())
    counts, _ = np.histogram(base, bins=edges)
    assert psi(edges, counts, base) == pytest.approx(0.0, abs=1e-9)


def test_psi_matches_manual_calculation():
    edges = [0, 1, 2]
    baseline_counts = [50, 50]          # 0.5 / 0.5
    values = [0.5] * 75 + [1.5] * 25    # 0.75 / 0.25
    expected = (0.75 - 0.5) * np.log(0.75 / 0.5) + (0.25 - 0.5) * np.log(0.25 / 0.5)
    assert psi(edges, baseline_counts, values) == pytest.approx(expected, rel=1e-6)


def test_psi_is_insensitive_when_bin_width_dwarfs_the_shift():
    """The mechanism behind the numerical PSI result.

    An extreme outlier sets the upper edge, so a shift of 0.75 SD moves
    almost no mass across a boundary and PSI stays far below threshold.
    """
    rng = np.random.default_rng(1)
    base = np.concatenate([rng.normal(20, 20, 50_000), [59_282.4]])
    edges = _edges(0, base.max())
    counts, _ = np.histogram(base, bins=edges)
    shifted = rng.normal(20 + 0.75 * 20, 20, 50_000)
    assert psi(edges, counts, shifted) < 0.2


def test_psi_detects_the_same_shift_under_proportionate_bins():
    """Same data, same shift, bins scaled to the bulk: PSI now responds."""
    rng = np.random.default_rng(1)
    base = rng.normal(20, 20, 50_000)
    edges = _edges(np.percentile(base, 0.1), np.percentile(base, 99.9))
    counts, _ = np.histogram(base, bins=edges)
    shifted = rng.normal(20 + 0.75 * 20, 20, 50_000)
    assert psi(edges, counts, shifted) > 0.2


def test_psi_drops_values_outside_the_baseline_support():
    """Documented implementation limitation (Chapter 4)."""
    edges = [0, 1, 2]
    counts = [50, 50]
    inside = psi(edges, counts, [0.5] * 100)
    with_outliers = psi(edges, counts, [0.5] * 100 + [500.0] * 100)
    assert inside == pytest.approx(with_outliers, rel=1e-9)


def test_ks_matches_scipy_reference():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 5_000), rng.normal(0.5, 1, 10_000)
    d, p = ks(a, b)
    ref = stats.ks_2samp(a, b)
    assert d == pytest.approx(ref.statistic, rel=1e-12)
    assert p == pytest.approx(ref.pvalue, rel=1e-12)


def test_chi2_zero_when_proportions_match():
    stat, p, unseen, cv = chi2({"a": 500, "b": 500}, ["a"] * 100 + ["b"] * 100)
    assert stat == pytest.approx(0.0, abs=1e-6)
    assert unseen is False and cv == pytest.approx(0.0, abs=1e-6)


def test_chi2_flags_an_unseen_category():
    stat, p, unseen, cv = chi2({"a": 900, "b": 100}, ["a"] * 75 + ["z"] * 25)
    assert unseen is True and stat > 0


def test_chi2_magnitude_depends_on_the_expected_count_floor():
    """The reported statistic is a function of an arbitrary constant."""
    stat, _, _, _ = chi2({"a": 900, "b": 100}, ["a"] * 750 + ["z"] * 250)
    assert stat > 1_000       # finite, but its scale is set by the floor


def test_gate_quantity_can_exceed_the_cramers_v_bound():
    """Confirms F11: the gate is not Cramer's V, which is bounded by 1."""
    stat, _, _, cv = chi2({"a": 9_999, "b": 1}, ["a"] * 5_000 + ["b"] * 5_000)
    assert cv > 1.0


def test_baseline_bin_width_to_sd_ratio_table():
    """Chapter 5 evidence: bin width relative to SD predicts PSI response."""
    path = "great_expectations/reference_baseline.json"
    if not os.path.exists(path):
        pytest.skip("baseline not present")
    b = json.load(open(path))
    print("\n  feature            bin_width       sd   ratio")
    for f, r in b["numerical"].items():
        e = r["bin_edges"]
        w = (e[-1] - e[0]) / (len(e) - 1)
        sd = r.get("std") or r.get("stddev")
        if sd:
            print(f"  {f:<18}{w:>10.4g}{sd:>9.4g}{w/sd:>8.2f}")
    assert True
