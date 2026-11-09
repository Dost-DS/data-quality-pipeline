"""Deterministic classification of drift alerts.

Every alert is assigned exactly one label, by the first rule that applies:

  1 CONFIRMED_DETECTION    the feature is an injected target and the window
                           starts at or after the recorded activation time
  2 EXPECTED_PROPAGATION   the feature is a declared dependent of an injected
                           feature. A deterministic edge suffices on its own;
                           an empirical edge additionally requires that the
                           same feature and metric did not alert in the control
  3 BASELINE_VARIATION     the same feature and metric alerted in the clean
                           control replay under an identical configuration
  4 UNEXPLAINED            none of the above 

Only UNEXPLAINED alerts count as false positives in the headline results.
"""
import yaml

DEPENDENCY_MAP = "great_expectations/dependency_map/feature_dependencies.yaml"

LABELS = ["CONFIRMED_DETECTION", "EXPECTED_PROPAGATION",
          "BASELINE_VARIATION", "UNEXPLAINED"]


def load_edges(path=DEPENDENCY_MAP):
    with open(path) as f:
        return yaml.safe_load(f)["edges"]


def propagation_targets(injected, edges):
    """Map each downstream feature to the edge that reaches it."""
    out = {}
    for e in edges:
        if e["source"] in injected and e["target"] not in injected:
            out.setdefault(e["target"], e)
    return out


def control_alert_set(control_rows):
    """(feature, metric) pairs that alerted in the clean control replay."""
    return {(r["feature_name"], r["metric_name"])
            for r in control_rows if r["is_drift"]}


def classify(feature, metric, window_start, injected, activation_ts,
             prop_targets, control_alerts):
    """Return (label, evidence) for a single alert."""
    after_activation = (activation_ts is None or window_start is None
                        or window_start >= activation_ts)

    # An alert on an injected feature is a detection only once the injection
    # is live. Before activation the data is identical to the control, so the
    # alert falls through to the checks below rather than being credited.
    if feature in injected and after_activation:
        return "CONFIRMED_DETECTION", {
            "rule": "injected target, window at or after activation",
            "feature": feature,
            "activation_timestamp": str(activation_ts),
            "window_start": str(window_start)}

    # Propagation likewise requires the upstream change to have occurred.
    if feature in prop_targets and after_activation:
        edge = prop_targets[feature]
        if edge["type"] == "deterministic":
            return "EXPECTED_PROPAGATION", {
                "rule": "deterministic dependency edge",
                "source": edge["source"], "target": edge["target"],
                "edge_type": edge["type"]}
        if (feature, metric) not in control_alerts:
            return "EXPECTED_PROPAGATION", {
                "rule": "empirical dependency edge, absent from control",
                "source": edge["source"], "target": edge["target"],
                "edge_type": edge["type"]}
        return "BASELINE_VARIATION", {
            "rule": "empirical edge, but the control also alerted here",
            "feature": feature, "metric": metric}

    if (feature, metric) in control_alerts:
        return "BASELINE_VARIATION", {
            "rule": "feature and metric alerted in the clean control",
            "feature": feature, "metric": metric}

    return "UNEXPLAINED", {
        "rule": "no injection, no dependency edge, absent from control",
        "feature": feature, "metric": metric}
