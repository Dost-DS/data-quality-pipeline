"""Regenerate every result table from the database. 

Until now several CSVs in results/ were produced ad hoc and had no writer in
the repository. The figures rendered only because the files were committed,
so a clean clone could not rebuild them. This script closes that gap: it
regenerates each table from dq_monitoring in one pass, covering every
experiment run present.

Column names are preserved exactly, because generate_figures.py reads
recall_by_metric.csv and pilot_window_size.csv by column name.

Definitions used throughout:
  post-activation   window_end >= activation_timestamp. A window can only
                    raise an alert once it closes, so window_end is the
                    correct boundary for detection and latency. The alert
                    classifier uses window_start instead, which is why its
                    per-scenario counts are 2,053 rather than 2,060.
  mean_pre          the clean control's pre-activation mean for the same
                    feature and metric. The scenario's own pre-activation
                    windows are not used, because a small number of windows
                    carry corrupted timestamps that place them before the
                    activation point while containing post-activation
                    records (see Chapter 4).
"""
import os

import pandas as pd
import psycopg2

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")
OUT = "results"


def q(sql):
    conn = psycopg2.connect(**PG)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def write(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"  {name:<32} {len(df):>5} rows")


ACT = """(SELECT MIN(activation_timestamp) FROM dq_monitoring.experiments
          WHERE activation_timestamp IS NOT NULL)"""


def main():
    os.makedirs(OUT, exist_ok=True)
    print("regenerating result tables from dq_monitoring\n")

    # ---- per-scenario drift summary --------------------------------------
    drift = q("""
        SELECT LEFT(d.experiment_id,2) AS scenario, d.feature_name,
               d.metric_name, d.feature_type,
               COUNT(*) AS scores,
               COUNT(*) FILTER (WHERE d.is_drift) AS alerts,
               ROUND(100.0*COUNT(*) FILTER (WHERE d.is_drift)/COUNT(*), 2) AS alert_pct,
               AVG(d.metric_value) AS mean_value,
               MAX(d.metric_value) AS max_value
        FROM dq_monitoring.drift_scores d
        JOIN dq_monitoring.experiments e USING (experiment_id)
        WHERE e.scenario <> 'PILOT'
        GROUP BY 1,2,3,4 ORDER BY 1, 6 DESC""")
    write(drift, "drift_by_scenario.csv")

    c0 = drift[drift.scenario == "C0"].drop(columns=["scenario"])
    write(c0.sort_values("alerts", ascending=False), "c0_control_alerts.csv")

    # ---- detection recall per injected target ----------------------------
    recall = q(f"""
        WITH tgt AS (
          SELECT e.experiment_id, LEFT(e.experiment_id,2) AS scenario,
                 jsonb_array_elements_text(
                     (e.ground_truth->'injected_targets')::jsonb) AS injected_feature
          FROM dq_monitoring.experiments e
          WHERE e.scenario NOT IN ('PILOT','C0')),
        ctrl AS (
          SELECT d.feature_name, d.metric_name, AVG(d.metric_value) AS mean_pre
          FROM dq_monitoring.drift_scores d
          JOIN dq_monitoring.experiments e USING (experiment_id)
          WHERE e.scenario = 'C0' AND d.window_end < {ACT}
          GROUP BY 1,2)
        SELECT t.scenario, t.injected_feature, d.metric_name,
               COUNT(*) FILTER (WHERE d.window_end >= {ACT}) AS post_windows,
               COUNT(*) FILTER (WHERE d.is_drift AND d.window_end >= {ACT}) AS post_alerts,
               AVG(d.metric_value) FILTER (WHERE d.window_end >= {ACT}) AS mean_post,
               MAX(c.mean_pre) AS mean_pre
        FROM tgt t
        JOIN dq_monitoring.drift_scores d
          ON d.experiment_id = t.experiment_id
         AND d.feature_name  = t.injected_feature
        LEFT JOIN ctrl c ON c.feature_name = d.feature_name
                        AND c.metric_name  = d.metric_name
        GROUP BY 1,2,3 ORDER BY 1,2,3""")
    write(recall, "recall_by_metric.csv")

    # ---- latency ---------------------------------------------------------
    lat = q(f"""
        WITH act AS (
          SELECT experiment_id, LEFT(experiment_id,2) AS scen,
                 activation_timestamp AS ts, ground_truth::text AS gt
          FROM dq_monitoring.experiments
          WHERE activation_timestamp IS NOT NULL AND scenario <> 'PILOT'),
        floors AS (
          SELECT a.scen, a.ts, MIN(d.window_end) AS floor_end
          FROM dq_monitoring.drift_scores d JOIN act a USING (experiment_id)
          WHERE d.window_end >= a.ts GROUP BY 1,2),
        agg AS (
          SELECT a.scen, d.feature_name, d.metric_name, a.ts, f.floor_end,
                 (a.gt LIKE '%"' || d.feature_name || '"%') AS is_injected_target,
                 COUNT(*) FILTER (WHERE d.window_end >= a.ts) AS post_windows,
                 COUNT(*) FILTER (WHERE d.is_drift AND d.window_end >= a.ts) AS post_alerts,
                 MIN(d.window_end) FILTER (WHERE d.is_drift
                     AND d.window_end >= a.ts) AS first_alert_end
          FROM dq_monitoring.drift_scores d
          JOIN act a USING (experiment_id)
          JOIN floors f ON f.scen = a.scen
          GROUP BY 1,2,3,4,5,6)
        SELECT scen AS scenario, feature_name, metric_name, is_injected_target,
               post_windows, post_alerts, first_alert_end, ts AS activation,
               floor_end - ts              AS window_close_interval,
               first_alert_end - ts        AS latency,
               first_alert_end - floor_end AS beyond_floor
        FROM agg ORDER BY 1,2,3""")
    write(lat, "latency_by_scenario.csv")

    # ---- classification and framework ------------------------------------
    cls = q("""
        SELECT LEFT(experiment_id,2) AS scenario, label, COUNT(*) AS alerts,
               ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (
                   PARTITION BY LEFT(experiment_id,2)), 2) AS pct_of_scenario
        FROM dq_monitoring.alert_classifications
        GROUP BY 1,2 ORDER BY 1,2""")
    write(cls, "alert_classification.csv")

    acts = q("""
        SELECT LEFT(ac.experiment_id,2) AS scenario, ac.recommended_action,
               ac.metric_name, COUNT(*) AS alerts
        FROM dq_monitoring.alert_classifications ac
        GROUP BY 1,2,3 ORDER BY 1,2,3""")
    write(acts, "framework_actions.csv")

    detail = q("""
        SELECT LEFT(ac.experiment_id,2) AS scenario, d.feature_name,
               ac.metric_name, ac.label, ac.recommended_action,
               COUNT(*) AS alerts, AVG(d.metric_value) AS mean_value
        FROM dq_monitoring.alert_classifications ac
        JOIN dq_monitoring.drift_scores d ON ac.drift_score_id = d.id
        GROUP BY 1,2,3,4,5 ORDER BY 1, 6 DESC""")
    write(detail, "framework_detail.csv")

    # ---- window-size pilot ----------------------------------------------
    pilot = q("""
        SELECT (e.ground_truth->>'window_size')::int AS window_size,
               d.metric_name, d.feature_type,
               COUNT(*) AS scores,
               COUNT(*) FILTER (WHERE d.is_drift) AS alerts,
               ROUND(100.0*COUNT(*) FILTER (WHERE d.is_drift)/COUNT(*), 2) AS alert_pct
        FROM dq_monitoring.drift_scores d
        JOIN dq_monitoring.experiments e USING (experiment_id)
        WHERE e.scenario = 'PILOT'
        GROUP BY 1,2,3 ORDER BY 1,2,3""")
    write(pilot, "pilot_window_size.csv")

    print(f"\ntotal alerts classified: "
          f"{q('SELECT COUNT(*) n FROM dq_monitoring.alert_classifications').n.iloc[0]:,}")


if __name__ == "__main__":
    main()
