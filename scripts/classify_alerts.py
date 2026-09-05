"""Apply the alert classification rule to every alert in every experiment.

Loads the clean control's alert set as the baseline reference, then labels
each alert in every scenario according to the fixed precedence in Table 3.2.
Results and their supporting evidence are written to alert_classifications.
"""
import os
import sys

import psycopg2
import psycopg2.extras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alert_classifier import load_edges, propagation_targets, classify

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")

CONTROL_SCENARIO = "C0"


def main():
    conn = psycopg2.connect(**PG)
    edges = load_edges()
    print(f"dependency map: {len(edges)} edges")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT experiment_id, scenario, activation_timestamp, ground_truth
                       FROM dq_monitoring.experiments ORDER BY scenario""")
        experiments = cur.fetchall()

        # The control's alert set is the reference for BASELINE_VARIATION.
        cur.execute("""SELECT DISTINCT d.feature_name, d.metric_name
                       FROM dq_monitoring.drift_scores d
                       JOIN dq_monitoring.experiments e USING (experiment_id)
                       WHERE e.scenario = %s AND d.is_drift""", (CONTROL_SCENARIO,))
        control_alerts = {(r["feature_name"], r["metric_name"]) for r in cur.fetchall()}

    print(f"control alert set: {len(control_alerts)} feature/metric pairs\n")

    # Clear any previous classification so the script is idempotent.
    with conn.cursor() as cur:
        cur.execute("TRUNCATE dq_monitoring.alert_classifications RESTART IDENTITY")
    conn.commit()

    totals = {}

    for exp in experiments:
        eid = exp["experiment_id"]
        scenario = exp["scenario"]
        gt = exp["ground_truth"]
        act = exp["activation_timestamp"]
        injected = set(gt.get("injected_targets") or [])
        prop = propagation_targets(injected, edges)

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""SELECT id, feature_name, metric_name, window_start
                           FROM dq_monitoring.drift_scores
                           WHERE experiment_id = %s AND is_drift
                           ORDER BY id""", (eid,))
            alerts = cur.fetchall()

        rows, counts = [], {}
        for a in alerts:
            label, ev = classify(a["feature_name"], a["metric_name"],
                                 a["window_start"], injected, act, prop, control_alerts)
            rows.append((eid, a["id"], a["feature_name"], a["metric_name"],
                         label, psycopg2.extras.Json(ev)))
            counts[label] = counts.get(label, 0) + 1
            totals[label] = totals.get(label, 0) + 1

        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur,
                """INSERT INTO dq_monitoring.alert_classifications
                   (experiment_id, drift_score_id, feature_name, metric_name, label, evidence)
                   VALUES (%s,%s,%s,%s,%s,%s)""", rows, page_size=1000)
        conn.commit()

        print(f"{scenario}  targets={sorted(injected) or '[]'}")
        print(f"    {len(alerts):>6,} alerts  ->  " +
              "  ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))

    conn.close()
    print(f"\ntotal: {sum(totals.values()):,} alerts classified")
    for k, v in sorted(totals.items()):
        print(f"  {k:<22} {v:>7,}  ({100*v/sum(totals.values()):.1f}%)")


if __name__ == "__main__":
    main()
