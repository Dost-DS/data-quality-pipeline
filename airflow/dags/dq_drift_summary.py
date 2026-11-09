"""Daily drift summary over stored monitoring results.

Aggregates drift_scores and alert_classifications per experiment and
reports alert rates by feature and metric, each against the clean control
baseline. Reads only; the experiments themselves are complete, so this
DAG summarises the metric store rather than driving new runs.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task

OUTPUT_CANDIDATES = ["/opt/airflow/results", "/opt/airflow/dags/../results",
                     "/opt/airflow/logs/dq_summary"]

DEFAULT_ARGS = {
    "owner": "dq-pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _connect():
    import psycopg2
    import psycopg2.extras
    return psycopg2.connect(host="postgres", port=5432, dbname="postgres",
                            user="pipeline_admin", password="pipeline_pass",
                            connect_timeout=10)


@dag(
    dag_id="dq_drift_summary",
    description="Daily alert-rate summary per experiment, feature and metric",
    start_date=datetime(2026, 9, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["thesis", "drift"],
)
def dq_drift_summary():

    @task
    def alert_rates() -> list[dict]:
        import psycopg2.extras
        conn = _connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT experiment_id, feature_name, metric_name,
                       COUNT(*) AS scores,
                       COUNT(*) FILTER (WHERE is_drift) AS alerts,
                       ROUND(100.0 * COUNT(*) FILTER (WHERE is_drift)
                             / NULLIF(COUNT(*), 0), 3) AS alert_pct
                FROM dq_monitoring.drift_scores
                GROUP BY 1, 2, 3
                ORDER BY 1, 5 DESC""")
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        print(f"summarised {len(rows)} experiment/feature/metric combinations")
        return rows

    @task
    def deltas_against_control(rows: list[dict]) -> list[dict]:
        control = {(r["feature_name"], r["metric_name"]): r["alerts"]
                   for r in rows if r["experiment_id"].startswith("C0")}
        out = []
        for r in rows:
            if r["experiment_id"].startswith("C0"):
                continue
            base = control.get((r["feature_name"], r["metric_name"]))
            if base is None:
                continue
            delta = r["alerts"] - base
            if delta != 0:
                out.append({"experiment_id": r["experiment_id"],
                            "feature_name": r["feature_name"],
                            "metric_name": r["metric_name"],
                            "control_alerts": base,
                            "scenario_alerts": r["alerts"],
                            "delta": delta})
        for d in sorted(out, key=lambda x: -abs(x["delta"])):
            print(f"  {d['experiment_id'][:2]} {d['feature_name']}/"
                  f"{d['metric_name']}: {d['control_alerts']} -> "
                  f"{d['scenario_alerts']} (delta {d['delta']:+,})")
        if not out:
            print("no feature/metric differs from the control")
        return out

    @task
    def classification_mix() -> list[dict]:
        import psycopg2.extras
        conn = _connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT label, recommended_action, COUNT(*) AS n
                FROM dq_monitoring.alert_classifications
                GROUP BY 1, 2 ORDER BY 3 DESC""")
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        total = sum(r["n"] for r in rows) or 1
        for r in rows:
            print(f"  {r['label']:<22}{str(r['recommended_action']):<14}"
                  f"{r['n']:>8,}  ({100*r['n']/total:.1f}%)")
        return rows

    @task
    def persist(rates: list[dict], deltas: list[dict], mix: list[dict]) -> str:
        target = None
        for cand in OUTPUT_CANDIDATES:
            try:
                os.makedirs(cand, exist_ok=True)
                probe = os.path.join(cand, ".write_probe")
                with open(probe, "w") as f:
                    f.write("ok")
                os.remove(probe)
                target = cand
                break
            except OSError:
                continue
        if target is None:
            print("no writable output directory; summary is in the task log only")
            return "logged"

        stamp = datetime.utcnow().strftime("%Y%m%d")
        for name, rows in (("alert_rates", rates), ("control_deltas", deltas),
                           ("classification_mix", mix)):
            if not rows:
                continue
            path = os.path.join(target, f"dag_{name}_{stamp}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {path} ({len(rows)} rows)")
        return target

    r = alert_rates()
    persist(r, deltas_against_control(r), classification_mix())


dq_drift_summary()
