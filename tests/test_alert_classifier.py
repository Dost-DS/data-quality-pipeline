import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from alert_classifier import (  # noqa: E402
    load_edges, propagation_targets, control_alert_set, classify, LABELS,
)

ACT = pd.Timestamp("2024-02-01 08:00:00")
AFTER = pd.Timestamp("2024-02-01 09:00:00")
BEFORE = pd.Timestamp("2024-02-01 07:00:00")


@pytest.fixture
def edges():
    return load_edges()


# --- the dependency map itself ---

def test_map_loads_and_is_typed(edges):
    assert len(edges) == 11
    assert all(e["type"] in ("deterministic", "empirical") for e in edges)
    assert all({"source", "target", "type", "description"} <= set(e) for e in edges)


def test_fare_to_total_is_deterministic(edges):
    e = next(x for x in edges
             if x["source"] == "fare_amount" and x["target"] == "total_amount")
    assert e["type"] == "deterministic"


def test_payment_to_tip_is_empirical(edges):
    e = next(x for x in edges
             if x["source"] == "payment_type" and x["target"] == "tip_amount")
    assert e["type"] == "empirical"


# --- each label in isolation ---

def test_confirmed_detection(edges):
    prop = propagation_targets({"fare_amount"}, edges)
    label, _ = classify("fare_amount", "ks", AFTER, {"fare_amount"}, ACT, prop, set())
    assert label == "CONFIRMED_DETECTION"


def test_expected_propagation_deterministic(edges):
    prop = propagation_targets({"fare_amount"}, edges)
    label, ev = classify("total_amount", "ks", AFTER, {"fare_amount"}, ACT, prop, set())
    assert label == "EXPECTED_PROPAGATION"
    assert ev["edge_type"] == "deterministic"


def test_expected_propagation_empirical(edges):
    prop = propagation_targets({"payment_type"}, edges)
    label, ev = classify("tip_amount", "ks", AFTER, {"payment_type"}, ACT, prop, set())
    assert label == "EXPECTED_PROPAGATION"
    assert ev["edge_type"] == "empirical"


def test_baseline_variation(edges):
    prop = propagation_targets({"fare_amount"}, edges)
    ctrl = {("PULocationID", "psi")}
    label, _ = classify("PULocationID", "psi", AFTER, {"fare_amount"}, ACT, prop, ctrl)
    assert label == "BASELINE_VARIATION"


def test_unexplained(edges):
    prop = propagation_targets({"fare_amount"}, edges)
    label, _ = classify("DOLocationID", "psi", AFTER, {"fare_amount"}, ACT, prop, set())
    assert label == "UNEXPLAINED"


# --- precedence conflicts ---

def test_injection_beats_propagation(edges):
    """A feature that is both injected and a propagation target is a detection."""
    injected = {"fare_amount", "total_amount"}
    prop = propagation_targets(injected, edges)
    label, _ = classify("total_amount", "ks", AFTER, injected, ACT, prop, set())
    assert label == "CONFIRMED_DETECTION"


def test_injection_beats_baseline_variation(edges):
    prop = propagation_targets({"fare_amount"}, edges)
    ctrl = {("fare_amount", "ks")}
    label, _ = classify("fare_amount", "ks", AFTER, {"fare_amount"}, ACT, prop, ctrl)
    assert label == "CONFIRMED_DETECTION"


def test_deterministic_propagation_beats_baseline_variation(edges):
    """A deterministic edge is causal, so it outranks a control coincidence."""
    prop = propagation_targets({"fare_amount"}, edges)
    ctrl = {("total_amount", "ks")}
    label, ev = classify("total_amount", "ks", AFTER, {"fare_amount"}, ACT, prop, ctrl)
    assert label == "EXPECTED_PROPAGATION"
    assert ev["edge_type"] == "deterministic"


def test_empirical_propagation_yields_to_baseline_variation(edges):
    """An empirical edge is only stochastic, so a control alert takes priority."""
    prop = propagation_targets({"payment_type"}, edges)
    ctrl = {("tip_amount", "ks")}
    label, _ = classify("tip_amount", "ks", AFTER, {"payment_type"}, ACT, prop, ctrl)
    assert label == "BASELINE_VARIATION"


# --- temporal edge cases ---

def test_alert_before_activation_is_unexplained(edges):
    prop = propagation_targets({"fare_amount"}, edges)
    label, _ = classify("fare_amount", "ks", BEFORE, {"fare_amount"}, ACT, prop, set())
    assert label == "UNEXPLAINED"


def test_alert_exactly_at_activation_counts(edges):
    prop = propagation_targets({"fare_amount"}, edges)
    label, _ = classify("fare_amount", "ks", ACT, {"fare_amount"}, ACT, prop, set())
    assert label == "CONFIRMED_DETECTION"


# --- control condition ---

def test_control_scenario_has_no_detections(edges):
    """With nothing injected, no alert can be a confirmed detection."""
    prop = propagation_targets(set(), edges)
    for feature in ["fare_amount", "total_amount", "PULocationID"]:
        label, _ = classify(feature, "ks", AFTER, set(), None, prop, set())
        assert label == "UNEXPLAINED"


def test_control_alert_set_filters_non_drift():
    rows = [
        {"feature_name": "a", "metric_name": "psi", "is_drift": True},
        {"feature_name": "b", "metric_name": "ks", "is_drift": False},
    ]
    assert control_alert_set(rows) == {("a", "psi")}


def test_every_label_is_declared():
    assert set(LABELS) == {"CONFIRMED_DETECTION", "EXPECTED_PROPAGATION",
                           "BASELINE_VARIATION", "UNEXPLAINED"}

