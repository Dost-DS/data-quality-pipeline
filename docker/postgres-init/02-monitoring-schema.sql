\c postgres

CREATE SCHEMA IF NOT EXISTS dq_monitoring;

-- One row per experimental run, with its ground truth
CREATE TABLE IF NOT EXISTS dq_monitoring.experiments (
    id                   SERIAL PRIMARY KEY,
    experiment_id        TEXT UNIQUE NOT NULL,
    scenario             TEXT NOT NULL,
    activation_idx       INTEGER,
    activation_timestamp TIMESTAMPTZ,
    ground_truth         JSONB NOT NULL,
    started_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Great Expectations results, one row per validated batch
CREATE TABLE IF NOT EXISTS dq_monitoring.validation_runs (
    id            SERIAL PRIMARY KEY,
    experiment_id TEXT,
    batch_id      TEXT NOT NULL,
    window_start  TIMESTAMPTZ,
    window_end    TIMESTAMPTZ,
    total_rows    INTEGER NOT NULL,
    passed_rows   INTEGER NOT NULL,
    failed_rows   INTEGER NOT NULL,
    success       BOOLEAN NOT NULL,
    run_time      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dq_monitoring.expectation_failures (
    id                SERIAL PRIMARY KEY,
    validation_run_id INTEGER REFERENCES dq_monitoring.validation_runs(id),
    expectation_type  TEXT NOT NULL,
    column_name       TEXT,
    failure_count     INTEGER NOT NULL,
    details           JSONB
);

CREATE TABLE IF NOT EXISTS dq_monitoring.quarantined_rows (
    id                  SERIAL PRIMARY KEY,
    validation_run_id   INTEGER REFERENCES dq_monitoring.validation_runs(id),
    row_data            JSONB NOT NULL,
    failed_expectations JSONB NOT NULL,
    quarantined_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per window x feature x metric
CREATE TABLE IF NOT EXISTS dq_monitoring.drift_scores (
    id            SERIAL PRIMARY KEY,
    experiment_id TEXT,
    window_id     TEXT NOT NULL,
    window_start  TIMESTAMPTZ,
    window_end    TIMESTAMPTZ,
    window_rows   INTEGER,
    feature_name  TEXT NOT NULL,
    feature_type  TEXT NOT NULL,
    metric_name   TEXT NOT NULL,
    metric_value  DOUBLE PRECISION,
    p_value       DOUBLE PRECISION,
    is_drift      BOOLEAN NOT NULL,
    computed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dq_monitoring.schema_snapshots (
    id                SERIAL PRIMARY KEY,
    experiment_id     TEXT,
    window_id         TEXT NOT NULL,
    window_start      TIMESTAMPTZ,
    window_end        TIMESTAMPTZ,
    new_columns       JSONB NOT NULL,
    missing_columns   JSONB NOT NULL,
    type_changes      JSONB NOT NULL,
    has_schema_change BOOLEAN NOT NULL,
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per alert, with its classification label and evidence
CREATE TABLE IF NOT EXISTS dq_monitoring.alert_classifications (
    id                 SERIAL PRIMARY KEY,
    experiment_id      TEXT NOT NULL,
    drift_score_id     INTEGER REFERENCES dq_monitoring.drift_scores(id),
    feature_name       TEXT NOT NULL,
    metric_name        TEXT NOT NULL,
    label              TEXT NOT NULL,
    evidence           JSONB NOT NULL,
    recommended_action TEXT,
    action_reason      TEXT,
    classified_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_validation_experiment ON dq_monitoring.validation_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_drift_experiment      ON dq_monitoring.drift_scores(experiment_id);
CREATE INDEX IF NOT EXISTS idx_drift_feature         ON dq_monitoring.drift_scores(feature_name);
CREATE INDEX IF NOT EXISTS idx_schema_experiment     ON dq_monitoring.schema_snapshots(experiment_id);
CREATE INDEX IF NOT EXISTS idx_alert_experiment      ON dq_monitoring.alert_classifications(experiment_id);
CREATE INDEX IF NOT EXISTS idx_alert_label           ON dq_monitoring.alert_classifications(label);
