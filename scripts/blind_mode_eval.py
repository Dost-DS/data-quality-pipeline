"""Blind-mode replication of D3/D4 (RQ4 evaluation).

Re-runs classification and interpretation with ground truth withheld:
no injected targets, no activation timestamp. Only the control replay
and the stored metric values are available, which is what an operator
has at runtime. Compares the resulting actions against the oracle.

Read-only with respect to the database.
"""
import csv
import os
import sys
from collections import Counter

import psycopg2
import psycopg2.extras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alert_classifier import classify, control_alert_set
from interpretation_framework import recommend

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")


def main():
    conn = psycopg2.connect(**PG)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT feature_name, metric_name, is_drift
                       FROM dq_monitoring.drift_scores
                       WHERE experiment_id LIKE 'C0%'""")
        control = control_alert_set(cur.fetchall())
    print(f"control alert pairs available to the blind operator: {len(control)}")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT ac.id, ac.experiment_id, ac.label AS oracle_label,
                              ac.recommended_action AS oracle_action,
                              d.feature_name, d.metric_name,
                              d.metric_value, d.window_start
                       FROM dq_monitoring.alert_classifications ac
                       JOIN dq_monitoring.drift_scores d
                         ON ac.drift_score_id = d.id
                       ORDER BY ac.id""")
        alerts = cur.fetchall()
    conn.close()
    print(f"alerts to re-interpret: {len(alerts):,}\n")

    crosstab, per_scen, rows = Counter(), {}, []
    for a in alerts:
        blind_label, _ = classify(
            feature=a["feature_name"], metric=a["metric_name"],
            window_start=a["window_start"],
            injected=frozenset(),
            activation_ts=None,
            prop_targets={},
            control_alerts=control)
        blind_action, _ = recommend(blind_label, a["metric_name"],
                                    a["metric_value"], schema_changed=False)
        crosstab[(a["oracle_action"], blind_action)] += 1
        scen = a["experiment_id"][:2]
        per_scen.setdefault(scen, Counter())[(a["oracle_action"], blind_action)] += 1
        rows.append((a["id"], scen, a["feature_name"], a["metric_name"],
                     a["oracle_label"], blind_label,
                     a["oracle_action"], blind_action))

    oracle_actions = sorted({k[0] for k in crosstab})
    blind_actions = sorted({k[1] for k in crosstab})

    print("=== oracle action (rows) x blind action (columns) ===")
    header = f"{'oracle':<14}" + "".join(f"{b:>13}" for b in blind_actions) + f"{'total':>13}"
    print(header)
    print("-" * len(header))
    for o in oracle_actions:
        cells = [crosstab.get((o, b), 0) for b in blind_actions]
        print(f"{o:<14}" + "".join(f"{c:>13,}" for c in cells) + f"{sum(cells):>13,}")

    print("\n=== surfacing rate: oracle ACT alerts not suppressed when blind ===")
    for scen in sorted(per_scen):
        ct = per_scen[scen]
        act = sum(v for (o, _), v in ct.items() if o == "ACT")
        kept = sum(v for (o, b), v in ct.items()
                   if o == "ACT" and b != "SUPPRESS")
        pct = f"{100*kept/act:.1f}%" if act else "n/a"
        print(f"  {scen}  oracle ACT {act:>6,}   surfaced when blind {kept:>6,}  ({pct})")

    os.makedirs("results", exist_ok=True)
    with open("results/blind_mode_detail.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alert_id", "scenario", "feature", "metric",
                    "oracle_label", "blind_label",
                    "oracle_action", "blind_action"])
        w.writerows(rows)
    with open("results/blind_mode_crosstab.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["oracle_action", "blind_action", "count"])
        for (o, b), n in sorted(crosstab.items()):
            w.writerow([o, b, n])
    print("\nwrote results/blind_mode_detail.csv and results/blind_mode_crosstab.csv")


if __name__ == "__main__":
    main()
