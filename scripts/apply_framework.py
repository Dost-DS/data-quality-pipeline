"""Apply the Alert Interpretation Framework to every classified alert (D4).""" 
import os
import sys

import psycopg2
import psycopg2.extras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interpretation_framework import recommend

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")


def main():
    conn = psycopg2.connect(**PG)

    # Windows in which a schema change was recorded, so criterion C7 can fire.
    with conn.cursor() as cur:
        cur.execute("""SELECT experiment_id, window_start
                       FROM dq_monitoring.schema_snapshots WHERE has_schema_change""")
        schema_windows = {(r[0], r[1]) for r in cur.fetchall()}
    print(f"windows with a schema change: {len(schema_windows)}")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT ac.id, ac.experiment_id, ac.label, ac.metric_name,
                              d.metric_value, d.window_start
                       FROM dq_monitoring.alert_classifications ac
                       JOIN dq_monitoring.drift_scores d ON ac.drift_score_id = d.id
                       ORDER BY ac.id""")
        alerts = cur.fetchall()

    print(f"alerts to interpret: {len(alerts):,}\n")

    updates, counts = [], {}
    for a in alerts:
        changed = (a["experiment_id"], a["window_start"]) in schema_windows
        action, reason = recommend(a["label"], a["metric_name"],
                                   a["metric_value"], schema_changed=changed)
        updates.append((action, reason, a["id"]))
        counts[action] = counts.get(action, 0) + 1

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur,
            """UPDATE dq_monitoring.alert_classifications
               SET recommended_action = %s, action_reason = %s WHERE id = %s""",
            updates, page_size=2000)
    conn.commit()
    conn.close()

    total = sum(counts.values())
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<12} {v:>7,}  ({100*v/total:.1f}%)")


if __name__ == "__main__":
    main()
