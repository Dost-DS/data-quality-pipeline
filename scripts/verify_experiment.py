"""Verify one completed experiment before proceeding to the next.""" 
import argparse
import psycopg2

PG = dict(host="localhost", port=5432, dbname="postgres",
          user="pipeline_admin", password="pipeline_pass")

QUERIES = [
    ("experiment registered",
     "SELECT COUNT(*) FROM dq_monitoring.experiments WHERE experiment_id = %s"),
    ("validation batches",
     "SELECT COUNT(*) FROM dq_monitoring.validation_runs WHERE experiment_id = %s"),
    ("drift windows",
     "SELECT COUNT(DISTINCT window_id) FROM dq_monitoring.drift_scores WHERE experiment_id = %s"),
    ("drift scores",
     "SELECT COUNT(*) FROM dq_monitoring.drift_scores WHERE experiment_id = %s"),
    ("drift alerts",
     "SELECT COUNT(*) FROM dq_monitoring.drift_scores WHERE experiment_id = %s AND is_drift"),
    ("schema windows",
     "SELECT COUNT(*) FROM dq_monitoring.schema_snapshots WHERE experiment_id = %s"),
]


def main(experiment_id):
    conn = psycopg2.connect(**PG)
    print(f"=== {experiment_id} ===")
    ok = True
    with conn.cursor() as cur:
        for label, q in QUERIES:
            cur.execute(q, (experiment_id,))
            n = cur.fetchone()[0]
            flag = "" if n > 0 else "   <-- EMPTY"
            if n == 0 and "alert" not in label:
                ok = False
            print(f"  {label:<22} {n:>10,}{flag}")

        cur.execute("""SELECT window_start, window_end, window_rows
                       FROM dq_monitoring.drift_scores WHERE experiment_id = %s
                       ORDER BY window_start LIMIT 1""", (experiment_id,))
        first = cur.fetchone()
        cur.execute("""SELECT window_start, window_end
                       FROM dq_monitoring.drift_scores WHERE experiment_id = %s
                       ORDER BY window_start DESC LIMIT 1""", (experiment_id,))
        last = cur.fetchone()
        cur.execute("""SELECT activation_timestamp, ground_truth->'injected_targets'
                       FROM dq_monitoring.experiments WHERE experiment_id = %s""",
                    (experiment_id,))
        act = cur.fetchone()

    if first and last and act:
        print(f"\n  first window   {first[0]} .. {first[1]} ({first[2]:,} rows)")
        print(f"  last window    {last[0]} .. {last[1]}")
        print(f"  activation     {act[0]}")
        print(f"  targets        {act[1]}")

    conn.close()
    print(f"\n  {'PASS' if ok else 'FAIL — a required table is empty'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_id")
    raise SystemExit(main(ap.parse_args().experiment_id))
