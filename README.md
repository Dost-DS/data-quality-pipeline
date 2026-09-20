# End-to-End Data Quality Monitoring Pipeline for Streaming Data

Bachelor's thesis artefact, B.Sc. Data Science, IU Internationale Hochschule.

Monitors a replayed stream of NYC TLC Yellow Taxi Trip Records (2024, ~41M records)
for data-quality defects, distributional drift, and schema change. Integrates
Apache Kafka, Great Expectations, PostgreSQL, MLflow, and Apache Airflow behind
five adapter interfaces, deployed with Docker Compose on a single node.

For the current project status, configuration rationale, experiment results, and a
running log of what changed and why, see `PROJECT_RECORD.md`. This file only covers
what's needed to set the pipeline up and run it.

## Architecture

A run is a full replay of the corpus, not a live deployment: each experiment deletes
and recreates the Kafka topic with a single partition, republishes all 41,169,720
records in event order, then runs validation, drift-detection, and schema-tracking
consumers over the replayed stream. Production and consumption are sequential.

`docker-compose.yml` defines seven services (Postgres, Zookeeper, Kafka, MLflow, and
three Airflow containers); six are long-running and `airflow-init` exits once the
Airflow metadata database is migrated.

Five points where a component can be substituted without changing the meaning of an
experiment, each backed by a second implementation and an equivalence test in
`tests/test_adapter_equivalence.py` / `tests/test_fallback_equivalence.py`:

| Interface   | Primary                      | Fallback              |
|-------------|-------------------------------|-------------------------|
| EventSource | Apache Kafka (single partition) | `ParquetEventSource`  |
| Validator   | Great Expectations 1.1.3        | `PlainValidator`      |
| MetricStore | PostgreSQL, schema `dq_monitoring` | `CsvMetricStore`  |
| RunTracker  | MLflow                          | `JsonRunTracker`      |
| Scheduler   | Apache Airflow                  | `SequentialScheduler` |

Both implementations of every boundary are declared as structural `Protocol`s in
`scripts/interfaces.py`, so the primary components conform without modification.

## Requirements

- WSL2 Ubuntu or native Linux -- do not run from a Windows-drive mount (`/mnt/*`);
  Docker volumes hold the database and that filesystem path is roughly ten times slower
- Python 3.11
- Docker and Docker Compose
- ~1 GB free disk for the raw Parquet corpus, plus room for the PostgreSQL volume

Dependencies are pinned in `requirements.txt` (14 direct packages) and
`requirements-lock.txt` (the full transitive closure).

## Setup

```bash
git clone <repo-url> data-quality-pipeline && cd data-quality-pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt

docker compose up -d
sleep 25   # let Postgres, Kafka and MLflow finish initialising

python scripts/download_dataset.py   # 12 monthly Parquet files into data/raw/
python scripts/init_gx_context.py    # Great Expectations file-backed context
python scripts/compute_baselines.py  # reference_baseline.json, schema_baseline.json
python scripts/define_expectations.py  # expectation suite + feature dependency map
```

## Running an experiment

Each scenario replay takes roughly three hours (produce ~104 min, validate ~30 min,
drift ~37 min, schema ~12 min), so launch it in `tmux` and let it run:

```bash
tmux new-session -d -s s1 "cd ~/thesis/data-quality-pipeline && source .venv/bin/activate && \
python scripts/run_experiment.py --scenario S1 --warmup-rows 20584860 --window-size 10000 \
--timeout-seconds 180 --report-every 2000000 2>&1 | tee logs/logs_S1.txt"
```

Scenarios: `C0` (clean control), `S1`-`S6` (distributional drift injections), `S7`
(schema change, no injection). Run `C0` first, and do not truncate `dq_monitoring`
between scenarios -- classification requires every earlier run to still be present.
Never launch the same scenario twice concurrently; a prior `tee` collision has
overwritten a run's log this way.

Run `python scripts/verify_experiment.py` between replays to confirm the five
monitoring tables were populated and to print the recorded activation point.

## Analysis

```bash
pytest tests/ -q                       # unit tests
python scripts/build_result_tables.py  # result CSVs from the database
python scripts/generate_figures.py     # figures from the database
python scripts/classify_alerts.py      # labels every alert (rebuilds all runs together)
python scripts/apply_framework.py      # recommended action per alert
python scripts/blind_mode_eval.py      # recall without ground truth
python scripts/blind_mode_v2.py        # percentile-threshold sweep
```

Integrity check after any restart:

```bash
docker exec -i pipeline-postgres psql -U pipeline_admin -d postgres -c \
  "SELECT COUNT(*) FROM dq_monitoring.alert_classifications;"
```

## Project layout

```
scripts/             pipeline code: producer/consumers, drift injection,
                      alert classification, interpretation framework, adapters,
                      fallback implementations, result and figure generation
tests/                unit tests (pytest)
airflow/dags/         DAGs demonstrating the Scheduler adapter
great_expectations/   expectation suite, reference baseline, schema baseline
data/raw/             downloaded Parquet corpus (not tracked)
figures/              generated figures -- regenerate with generate_figures.py
results/              generated CSV result tables
logs/                 per-run console logs
PROJECT_RECORD.md     current status, rationale, and a log of what changed and why
```
