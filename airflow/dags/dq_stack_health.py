"""Scheduled health check for the monitoring stack (pipeline step 19).

Verifies that each component the pipeline depends on is reachable and
that the metric store holds the schema the consumers write to. Runs
hourly and fails loudly rather than degrading silently, since a
monitoring system that is itself down produces no alerts and therefore
looks identical to a system with nothing to report.
"""
from __future__ import annotations

import socket
import urllib.request
from datetime import datetime, timedelta

from airflow.decorators import dag, task

REQUIRED_TABLES = [
    "experiments", "validation_runs", "expectation_failures",
    "quarantined_rows", "drift_scores", "schema_snapshots",
    "alert_classifications",
]

DEFAULT_ARGS = {
    "owner": "dq-pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="dq_stack_health",
    description="Hourly reachability and schema check for the monitoring stack",
    start_date=datetime(2026, 9, 1),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["thesis", "health"],
)
def dq_stack_health():

    @task
    def check_postgres() -> dict:
        import psycopg2
        conn = psycopg2.connect(host="postgres", port=5432, dbname="postgres",
                                user="pipeline_admin", password="pipeline_pass",
                                connect_timeout=10)
        with conn.cursor() as cur:
            cur.execute("""SELECT table_name FROM information_schema.tables
                           WHERE table_schema = 'dq_monitoring'""")
            found = {r[0] for r in cur.fetchall()}
        conn.close()
        missing = sorted(set(REQUIRED_TABLES) - found)
        if missing:
            raise RuntimeError(f"dq_monitoring is missing tables: {missing}")
        print(f"postgres reachable; all {len(REQUIRED_TABLES)} tables present")
        return {"tables": sorted(found)}

    @task
    def check_kafka() -> dict:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(bootstrap_servers="kafka:9092",
                                 consumer_timeout_ms=10_000,
                                 request_timeout_ms=10_000,
                                 api_version_auto_timeout_ms=10_000)
        topics = sorted(consumer.topics())
        consumer.close()
        print(f"kafka reachable; topics: {topics}")
        return {"topics": topics}

    @task
    def check_mlflow() -> dict:
        # The Airflow image has no curl, so use urllib (project record, sec. 9).
        last_error = None
        for host in ("mlflow", "pipeline-mlflow"):
            url = f"http://{host}:5000/health"
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    print(f"mlflow reachable at {url} (HTTP {r.status})")
                    return {"host": host, "status": r.status}
            except Exception as exc:
                last_error = f"{host}: {type(exc).__name__}"
        raise RuntimeError(f"mlflow unreachable ({last_error})")

    @task
    def check_zookeeper() -> dict:
        # 'srvr' is enabled by default; 'ruok' is not (project record, sec. 9).
        s = socket.create_connection(("zookeeper", 2181), timeout=10)
        s.sendall(b"srvr")
        reply = s.recv(2048).decode(errors="replace")
        s.close()
        if "Zookeeper version" not in reply:
            raise RuntimeError(f"unexpected srvr reply: {reply[:120]!r}")
        print("zookeeper reachable")
        return {"srvr": reply.splitlines()[0]}

    @task
    def report(pg: dict, kafka: dict, mlflow: dict, zk: dict) -> str:
        summary = (f"postgres tables={len(pg['tables'])} "
                   f"kafka topics={len(kafka['topics'])} "
                   f"mlflow={mlflow['status']} zookeeper=ok")
        print(f"stack healthy: {summary}")
        return summary

    report(check_postgres(), check_kafka(), check_mlflow(), check_zookeeper())


dq_stack_health()
