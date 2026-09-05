import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from interpretation_framework import (  # noqa: E402
    recommend, CRITERIA, ACTIONS, PSI_STRONG, KS_STRONG,
)


def test_propagation_is_accepted():
    action, _ = recommend("EXPECTED_PROPAGATION", "ks", 0.6)
    assert action == "ACCEPT"


def test_baseline_variation_is_suppressed():
    action, _ = recommend("BASELINE_VARIATION", "psi", 0.9)
    assert action == "SUPPRESS"


def test_strong_detection_acts():
    action, _ = recommend("CONFIRMED_DETECTION", "ks", KS_STRONG + 0.01)
    assert action == "ACT"
    action, _ = recommend("CONFIRMED_DETECTION", "psi", PSI_STRONG + 0.01)
    assert action == "ACT"


def test_marginal_detection_investigates():
    """A detection just over the alerting threshold is not yet actionable."""
    action, reason = recommend("CONFIRMED_DETECTION", "ks", KS_STRONG - 0.01)
    assert action == "INVESTIGATE"
    assert "practically significant" in reason


def test_chi2_detection_acts_without_magnitude_check():
    action, reason = recommend("CONFIRMED_DETECTION", "chi2", 78194.0)
    assert action == "ACT"
    assert "unbounded" in reason


def test_unexplained_strong_investigates():
    action, _ = recommend("UNEXPLAINED", "psi", PSI_STRONG + 0.1)
    assert action == "INVESTIGATE"


def test_unexplained_weak_monitors():
    action, _ = recommend("UNEXPLAINED", "psi", 0.25)
    assert action == "MONITOR"


def test_schema_change_overrides_everything():
    for label in ["CONFIRMED_DETECTION", "EXPECTED_PROPAGATION",
                  "BASELINE_VARIATION", "UNEXPLAINED"]:
        action, reason = recommend(label, "ks", 0.9, schema_changed=True)
        assert action == "INVESTIGATE"
        assert "schema" in reason


def test_missing_magnitude_is_not_strong():
    action, _ = recommend("CONFIRMED_DETECTION", "ks", None)
    assert action == "INVESTIGATE"


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        recommend("SOMETHING_ELSE", "ks", 0.5)


def test_checklist_is_complete():
    assert len(CRITERIA) == 7
    for c in CRITERIA:
        assert {"id", "name", "question", "evidence", "interpretation"} <= set(c)
    assert [c["id"] for c in CRITERIA] == [f"C{i}" for i in range(1, 8)]


def test_all_actions_reachable():
    produced = {
        recommend("CONFIRMED_DETECTION", "ks", 0.9)[0],
        recommend("CONFIRMED_DETECTION", "ks", 0.15)[0],
        recommend("EXPECTED_PROPAGATION", "ks", 0.5)[0],
        recommend("BASELINE_VARIATION", "ks", 0.5)[0],
        recommend("UNEXPLAINED", "psi", 0.25)[0],
    }
    assert produced == set(ACTIONS)
