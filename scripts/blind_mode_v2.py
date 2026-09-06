"""Blind mode with a magnitude-aware baseline comparison (revised C3).

The operator is assumed to hold the control replay's full distribution of
metric values per (feature, metric), not merely a binary alert flag. An
alert is suppressed only if its magnitude falls within that distribution.
Ground truth remains withheld. Read-only.
"""
import csv
import os
from collections import Counter

import numpy as np
import psycopg2
import psycopg2.extras

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")

PSI_STRONG, KS_STRONG = 0.4, 0.2
PERCENTILES = [95.0, 99.0, 99.9, 100.0]


def strong(metric, value):
    if metric == "psi":
        return value >= PSI_STRONG
    if metric == "ks":
        return value >= KS_STRONG
    return False          # chi2 magnitude is not assessable


def main():
    conn = psycopg2.connect(**PG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT feature_name, metric_name, metric_value
                       FROM dq_monitoring.drift_scores
                       WHERE experiment_id LIKE 'C0%' AND metric_value IS NOT NULL""")
        ctrl = {}
        for r in cur.fetchall():
            ctrl.setdefault((r["feature_name"], r["metric_name"]), []).append(
                float(r["metric_value"]))

        cur.execute("""SELECT ac.experiment_id, ac.recommended_action AS oracle,
                              d.feature_name, d.metric_name, d.metric_value
                       FROM dq_monitoring.alert_classifications ac
                       JOIN dq_monitoring.drift_scores d ON ac.drift_score_id = d.id
                       WHERE ac.experiment_id NOT LIKE 'C0%'""")
        alerts = cur.fetchall()
    conn.close()

    cuts = {k: {p: float(np.percentile(v, p)) for p in PERCENTILES}
            for k, v in ctrl.items()}

    print("=== control distribution of metric values (C0, all windows) ===")
    print(f"{'feature':<20}{'metric':<7}{'p95':>12}{'p99':>12}{'p99.9':>12}{'max':>12}")
    for (f, m), c in sorted(cuts.items()):
        print(f"{f:<20}{m:<7}" + "".join(f"{c[p]:>12.4g}" for p in PERCENTILES))

    rows = []
    for p in PERCENTILES:
        per_scen, kept_tot, act_tot = {}, 0, 0
        for a in alerts:
            key = (a["feature_name"], a["metric_name"])
            val = float(a["metric_value"]) if a["metric_value"] is not None else 0.0
            escalated = key in cuts and val > cuts[key][p]
            action = ("ACT" if strong(a["metric_name"], val) else "INVESTIGATE") \
                if escalated else "SUPPRESS"
            s = a["experiment_id"][:2]
            d = per_scen.setdefault(s, Counter())
            d["total"] += 1
            d[action] += 1
            if a["oracle"] == "ACT":
                d["oracle_act"] += 1
                act_tot += 1
                if action != "SUPPRESS":
                    d["surfaced"] += 1
                    kept_tot += 1
            if a["oracle"] == "SUPPRESS" and action != "SUPPRESS":
                d["noise_leaked"] += 1

        print(f"\n=== percentile cut p{p} ===")
        print(f"{'scen':<6}{'alerts':>9}{'oracleACT':>11}{'surfaced':>10}"
              f"{'recall':>9}{'noiseLeak':>11}{'shown':>9}")
        for s in sorted(per_scen):
            d = per_scen[s]
            shown = d["total"] - d["SUPPRESS"]
            rec = 100 * d["surfaced"] / d["oracle_act"] if d["oracle_act"] else 0
            print(f"{s:<6}{d['total']:>9,}{d['oracle_act']:>11,}{d['surfaced']:>10,}"
                  f"{rec:>8.1f}%{d['noise_leaked']:>11,}{shown:>9,}")
            rows.append([p, s, d["total"], d["oracle_act"], d["surfaced"],
                         round(rec, 2), d["noise_leaked"], shown])
        print(f"  overall detection recall: "
              f"{100*kept_tot/act_tot if act_tot else 0:.1f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/blind_mode_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["percentile", "scenario", "alerts", "oracle_act",
                    "surfaced", "recall_pct", "noise_leaked", "shown_to_operator"])
        w.writerows(rows)
    print("\nwrote results/blind_mode_v2.csv")


if __name__ == "__main__":
    main()
