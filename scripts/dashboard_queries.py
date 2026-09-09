"""Queries backing the monitoring dashboard.

Every figure shown in the dashboard is read from the experiment database,
so nothing displayed is synthetic or hand-entered.
"""
import os
import psycopg2
import psycopg2.extras

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")

SCENARIO_LABELS = {
    "C0": "Clean control",
    "S1": "Numerical mean shift",
    "S2": "Categorical value swap",
    "S3": "Unseen category",
    "S4": "Numerical variance shift",
    "S5": "Mixed change",
}

SCENARIO_TARGETS = {
    "C0": [], "S1": ["fare_amount"], "S2": ["payment_type"],
    "S3": ["RatecodeID"], "S4": ["trip_distance"],
    "S5": ["fare_amount", "payment_type"],
}


def _rows(cur, sql, args=None):
    cur.execute(sql, args or ())
    return [dict(r) for r in cur.fetchall()]


def collect():
    conn = psycopg2.connect(**PG)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Records per replay, not the sum across replays: each experiment
    # streams the same corpus, so summing would multiply it by the run count.
    corpus = _rows(cur, """
        SELECT COUNT(DISTINCT experiment_id) AS experiments,
               MAX(rows_per_run) AS rows_processed
        FROM (SELECT experiment_id, SUM(total_rows) AS rows_per_run
              FROM dq_monitoring.validation_runs
              GROUP BY experiment_id) t""")[0]

    windows = _rows(cur, """
        SELECT COUNT(*) AS n FROM dq_monitoring.drift_scores""")[0]

    activation = _rows(cur, """
        SELECT MIN(activation_timestamp) AS ts FROM dq_monitoring.experiments""")[0]

    scenarios = _rows(cur, """
        SELECT e.scenario,
               COUNT(*) AS scores,
               COUNT(*) FILTER (WHERE d.is_drift) AS alerts,
               COUNT(DISTINCT d.window_id) AS windows
        FROM dq_monitoring.drift_scores d
        JOIN dq_monitoring.experiments e USING (experiment_id)
        GROUP BY 1 ORDER BY 1""")

    classification = _rows(cur, """
        SELECT e.scenario, ac.label, COUNT(*) AS n
        FROM dq_monitoring.alert_classifications ac
        JOIN dq_monitoring.experiments e USING (experiment_id)
        GROUP BY 1,2 ORDER BY 1,2""")

    actions = _rows(cur, """
        SELECT e.scenario, ac.recommended_action AS action, COUNT(*) AS n
        FROM dq_monitoring.alert_classifications ac
        JOIN dq_monitoring.experiments e USING (experiment_id)
        WHERE ac.recommended_action IS NOT NULL
        GROUP BY 1,2 ORDER BY 1,2""")

    # Daily alert density. Windows carrying corrupted timestamps are excluded
    # so that the timeline reflects the study year rather than meter faults.
    timeline = _rows(cur, """
        SELECT e.scenario,
               date_trunc('day', d.window_start)::date AS day,
               COUNT(*) FILTER (WHERE d.is_drift) AS alerts
        FROM dq_monitoring.drift_scores d
        JOIN dq_monitoring.experiments e USING (experiment_id)
        WHERE d.window_start >= '2024-01-01' AND d.window_start < '2025-01-01'
        GROUP BY 1,2 ORDER BY 1,2""")

    per_feature = _rows(cur, """
        SELECT e.scenario, d.feature_name, d.feature_type, d.metric_name,
               COUNT(*) AS scores,
               COUNT(*) FILTER (WHERE d.is_drift) AS alerts,
               ROUND(AVG(d.metric_value)::numeric, 6)::float8 AS mean_value
        FROM dq_monitoring.drift_scores d
        JOIN dq_monitoring.experiments e USING (experiment_id)
        GROUP BY 1,2,3,4 ORDER BY 1,2,4""")

    recall = _rows(cur, """
        SELECT e.scenario, gt.target AS feature_name, d.metric_name,
               COUNT(*) FILTER (WHERE d.is_drift
                    AND d.window_end >= e.activation_timestamp) AS post_alerts,
               ROUND(AVG(d.metric_value) FILTER (
                    WHERE d.window_end >= e.activation_timestamp)::numeric,6)::float8 AS mean_post,
               ROUND(AVG(d.metric_value) FILTER (
                    WHERE d.window_end < e.activation_timestamp)::numeric,6)::float8 AS mean_pre
        FROM dq_monitoring.experiments e
        CROSS JOIN LATERAL jsonb_array_elements_text(
            e.ground_truth->'injected_targets') AS gt(target)
        JOIN dq_monitoring.drift_scores d
          ON d.experiment_id = e.experiment_id AND d.feature_name = gt.target
        WHERE e.scenario <> 'C0'
        GROUP BY 1,2,3 ORDER BY 1,2,3""")

    validation = _rows(cur, """
        SELECT e.scenario,
               SUM(vr.total_rows) AS rows_checked,
               SUM(vr.failed_rows) AS rows_failed
        FROM dq_monitoring.validation_runs vr
        JOIN dq_monitoring.experiments e USING (experiment_id)
        GROUP BY 1 ORDER BY 1""")

    cur.close()
    conn.close()

    return {
        "corpus": {
            "rows": int(corpus["rows_processed"] or 0),
            "experiments": int(corpus["experiments"] or 0),
            "scores": int(windows["n"] or 0),
            "activation": str(activation["ts"]),
        },
        "labels": SCENARIO_LABELS,
        "targets": SCENARIO_TARGETS,
        "scenarios": scenarios,
        "classification": classification,
        "actions": actions,
        "timeline": [{**r, "day": str(r["day"])} for r in timeline],
        "per_feature": per_feature,
        "recall": recall,
        "validation": validation,
    }
