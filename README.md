# End-to-End Data Quality Monitoring Pipeline for Streaming Data

Bachelor's thesis artefact, B.Sc. Data Science, IU Internationale Hochschule.

Monitors a replayed stream of NYC TLC Yellow Taxi Trip Records (2024, ~41M records)
for data-quality defects, distributional drift, and schema change. Integrates
Apache Kafka, Great Expectations, PostgreSQL, MLflow, and Apache Airflow behind
five adapter interfaces, deployed with Docker Compose on a single node.

See `docs/` for the reproduction procedure.
