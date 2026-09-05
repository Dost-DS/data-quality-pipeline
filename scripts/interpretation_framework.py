"""Alert Interpretation Framework (deliverable D3).

Takes a classified alert together with its stored metric record and returns
a recommended action. The classification rule established what an alert is;
this decides what to do about it.

Actions:
  ACT         a confirmed detection with sufficient magnitude
  INVESTIGATE a confirmed detection whose magnitude is marginal, or an
              unexplained alert with strong evidence
  MONITOR     an unexplained alert with weak evidence
  ACCEPT      a change fully explained by a declared dependency
  SUPPRESS    an alert the clean control also produced
"""

PSI_THRESHOLD = 0.2
KS_MIN_EFFECT = 0.1
CHI2_MIN_CRAMERS_V = 0.1

# Magnitude at which a confirmed detection is unambiguous rather than marginal.
PSI_STRONG = 2 * PSI_THRESHOLD
KS_STRONG = 2 * KS_MIN_EFFECT

ACTIONS = ["ACT", "INVESTIGATE", "MONITOR", "ACCEPT", "SUPPRESS"]

# Deliverable D2: the operator-facing checklist.
CRITERIA = [
    {"id": "C1", "name": "Ground truth",
     "question": "Is the feature a declared injection target, and does the window "
                 "close at or after the activation time?",
     "evidence": "experiments.ground_truth.injected_targets, activation_timestamp, "
                 "drift_scores.window_end",
     "interpretation": "Yes: a confirmed detection. Record the latency and go to C4."},
    {"id": "C2", "name": "Dependency",
     "question": "Is the feature a declared dependent of a feature that has already "
                 "alerted, and is the edge deterministic?",
     "evidence": "feature_dependencies.yaml edge type and source",
     "interpretation": "Yes: expected propagation, not a detector error. Accept it."},
    {"id": "C3", "name": "Baseline comparison",
     "question": "Did the same feature and metric alert in the clean control replay?",
     "evidence": "drift_scores for the control experiment",
     "interpretation": "Yes: attributable to natural variation. Suppress it."},
    {"id": "C4", "name": "Practical significance",
     "question": "Does the magnitude clear twice the alerting threshold?",
     "evidence": "drift_scores.metric_value against PSI 0.4 or KS D 0.2",
     "interpretation": "Yes: act. No: the change is detectable but small, so investigate "
                       "before acting (Yurdakul & Naranjo, 2020)."},
    {"id": "C5", "name": "Metric disagreement",
     "question": "Does PSI stay below threshold in a window where KS or chi-square fires?",
     "evidence": "the PSI and KS or chi-square rows for the same window and feature",
     "interpretation": "Yes: check whether the binning is adequate for this feature "
                       "before trusting either verdict (Haas & Sibbald, 2024)."},
    {"id": "C6", "name": "Sample size",
     "question": "Is the window large enough that the p-value is driven by n rather "
                 "than by effect size?",
     "evidence": "drift_scores.window_rows alongside the effect size",
     "interpretation": "Read the KS D statistic or the PSI magnitude rather than the "
                       "p-value (Lu et al., 2025)."},
    {"id": "C7", "name": "Schema",
     "question": "Was a schema-change event recorded for the same window?",
     "evidence": "schema_snapshots.has_schema_change",
     "interpretation": "Yes: investigate the schema change first, as it is the more "
                       "likely cause of an apparent distributional shift."},
]


def _magnitude_is_strong(metric_name, metric_value):
    if metric_value is None:
        return False
    if metric_name == "psi":
        return metric_value >= PSI_STRONG
    if metric_name == "ks":
        return metric_value >= KS_STRONG
    if metric_name == "chi2":
        # The chi-square statistic has no interpretable scale once expected
        # counts approach zero, so magnitude alone cannot qualify it.
        return False
    return False


def recommend(label, metric_name, metric_value, schema_changed=False):
    """Return (action, reason) for one classified alert."""
    if schema_changed:
        return "INVESTIGATE", ("a schema change was recorded in the same window, "
                               "which is the more likely cause")

    if label == "EXPECTED_PROPAGATION":
        return "ACCEPT", "explained by a declared dependency on an injected feature"

    if label == "BASELINE_VARIATION":
        return "SUPPRESS", "the clean control replay produced the same alert"

    if label == "CONFIRMED_DETECTION":
        if metric_name == "chi2":
            return "ACT", ("confirmed detection; magnitude not assessable because the "
                           "chi-square scale is unbounded under sparse expected counts")
        if _magnitude_is_strong(metric_name, metric_value):
            return "ACT", "confirmed detection with magnitude beyond twice the threshold"
        return "INVESTIGATE", ("confirmed detection, but the magnitude is close to the "
                              "threshold and may not be practically significant")

    if label == "UNEXPLAINED":
        if _magnitude_is_strong(metric_name, metric_value):
            return "INVESTIGATE", "unexplained alert with a large magnitude"
        return "MONITOR", "unexplained alert with a marginal magnitude"

    raise ValueError(f"unknown label: {label!r}")
