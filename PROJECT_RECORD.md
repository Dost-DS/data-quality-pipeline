# Bachelor Thesis Project Record

**Designing and Evaluating an End-to-End Data Quality Monitoring Pipeline for Streaming Data**

Dost Muhammad, B.Sc. Data Science, IU International University of Applied Sciences
Supervisor: Dr. Mark Menzel. 

Setup and run instructions are in `README.md`.

---

## 1. Where the project stands

| Area | Status |
|---|---|
| Pipeline | Complete. All 20 build steps finished on 6 September |
| Experiments | Eight runs: C0 and S1 to S7, plus the window-size pilot |
| Alert classification | 108,665 alerts, zero unattributed |
| Framework | D1 to D4 complete, with an evaluation without ground truth |
| Tests | 88 |
| Figures | `generate_figures.py` produces 11 |
| Thesis | Complete: eight chapters, 76 references, appendices A to P |
| Repository | Published at https://github.com/Dost-DS/data-quality-pipeline |
| Remaining | Submission |

---

## 2. Environment

Windows 11, WSL2 Ubuntu 26.04, AMD Ryzen 9 9900X3D, 64 GB RAM.
WSL limited to 48 GB of memory and 12 processors.

The project runs in WSL. Do not move it to a Windows drive: Docker volumes hold the
database, and the `/mnt/*` file system is roughly ten times slower.

| Component | Version |
|---|---|
| Python | 3.11.9 (pyenv) |
| Docker / Compose | 29.6.1 / v5.3.0 |
| PostgreSQL | 16.15 |
| Kafka / ZooKeeper (Confluent) | 7.7.1 |
| Great Expectations | 1.1.3 (requires pandas below 2.2) |
| pandas / NumPy / SciPy | 2.1.4 / 1.26.4 / 1.13.1 |
| MLflow | 2.17.2 |
| Airflow | 2.10.5 |

92 dependencies are pinned in `requirements-lock.txt`.

---

## 3. Dataset

NYC TLC Yellow Taxi Trip Records 2024, twelve monthly Parquet files.
41,169,720 records, 660.9 MB, 19 columns. Reference week 1 to 7 January, 609,698 records.

Numerical (PSI and KS): `trip_distance`, `fare_amount`, `total_amount`,
`passenger_count`, `tip_amount`

Categorical (PSI and chi-square): `VendorID` (3), `payment_type` (5), `RatecodeID` (7),
`store_and_fwd_flag` (2), `PULocationID` (248), `DOLocationID` (257)

22 feature and metric pairs per window: 90,574 scores / 4,117 windows = 22.

### Reference bin geometry, the source of the PSI result

| Feature | Bin width / SD | Modal-bin share | Empty bins | Max PSI |
|---|---|---|---|---|
| passenger_count | 0.88 | 73.1895% | 1 of 10 | 0.2463 |
| tip_amount | 8.78 | 99.8740% | 2 of 10 | 0.0136 |
| total_amount | 9.38 | 98.6073% | 1 of 10 | 0.0262 |
| fare_amount | 11.58 | 98.8338% | 2 of 10 | 0.0223 |
| trip_distance | 69.38 | 99.9995% | 6 of 10 | 0.0069 |

The modal share orders the five features exactly inversely to their maximum PSI. The
bin-width ratio does not (`tip_amount` beats `fare_amount` on ratio but not on maximum),
so the thesis argues from concentration and uses the ratio as context.

The modal share is computed over non-null records. `passenger_count` has about 28,000
nulls in the reference week, so the share is 425,670 / 581,600 = 73.19%, not
425,670 / 609,698 = 69.82%. Recomputing it from `row_count` gives the wrong figure.

Twelve reference bins hold exactly zero records, the condition under which Haas and
Sibbald describe the index as undefined.

### Naturally occurring defects

15 backward-dated pickups (2002, 2008, 2009); forward-dated pickups up to 2026-06-26;
420 records (0.00102%) outside their file's nominal month. After sorting there are no
descents in event time, and 131 records overlap at month boundaries in 41 million.

---

## 4. Architecture

Seven services under Docker Compose, six long-running (`airflow-init` exits once the
Airflow metadata database is migrated). Replay is republication from source: each run
deletes and recreates the topic with one partition, then republishes. Production and
consumption are sequential, so the artifact is a replay harness, not a live deployment.

### Five adapter interfaces, each with a second implementation

| Interface | Primary | Fallback |
|---|---|---|
| EventSource | Kafka | `ParquetEventSource` |
| Validator | Great Expectations 1.1.3 | `PlainValidator` |
| MetricStore | PostgreSQL | `CsvMetricStore` |
| RunTracker | MLflow | `JsonRunTracker` |
| Scheduler | Airflow | `SequentialScheduler` |

No fallback was used in any experiment. Two caveats: both validators read the same suite
JSON, so the tests show that the engine can be substituted, not the rule specification;
and the MLflow calls used in the runs omit a step index.

### Database: schema `dq_monitoring`, 7 tables

`experiments`, `validation_runs`, `expectation_failures`, `quarantined_rows`,
`drift_scores`, `schema_snapshots`, `alert_classifications`.

Known gap: `drift_scores` stores `metric_value` and `p_value` but not the categorical
effect size or the per-window category counts. This is why the one non-alerting
`RatecodeID` window in S3 and S6 cannot be diagnosed. The thesis states this limitation
in Section 7.4. Add `cramers_v` and `n_categories` before any future run.

### Web interfaces

| Service | Address |
|---|---|
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8090 |

Ports from `docker compose ps`: airflow-webserver 8090->8080, mlflow 5000, postgres 5432,
kafka 29092. The Airflow scheduler and ZooKeeper publish no port.

---

## 5. Configuration

Window size 10,000, confirmed by a five-size pilot on a clean 200,000-row slice
(Table 5.1 of the thesis):

| Window | PSI (num) | PSI (cat) | KS | Chi-square |
|---|---|---|---|---|
| 1,000 | 0.60% | 33.58% | 26.50% | 9.25% |
| 2,500 | 0.50% | 32.50% | 23.25% | 7.29% |
| 5,000 | 0.00% | 23.75% | 20.50% | 6.25% |
| 10,000 | 0.00% | 8.33% | 17.00% | 5.00% |
| 25,000 | 0.00% | 0.00% | 17.50% | 2.08% |

PSI has to be split by feature type: aggregated, it looks like the quietest metric, when
its numerical half is simply silent. KS levels off while the others fall, so what remains
is real drift against a January reference.

```
PSI_THRESHOLD      = 0.2    # between Siddiqi's 0.10 and 0.25 bands (p. 137)
P_VALUE_THRESHOLD  = 0.05
MIN_WINDOW_ROWS    = 500
KS_MIN_EFFECT      = 0.1
CHI2_MIN_CRAMERS_V = 0.1    # not Cramer's V, see below
```

Siddiqi (2006, p. 137) does not give 0.2. He gives bands at 0.10 and 0.25, and 0.2 was
fixed between them before any results were seen. He also computes the index over score
bands, not raw feature bins, and states that it is not an end-all measure.

The categorical gate is not Cramer's V. It computes sqrt(chi2 / (n(k-1))). Cramer's bound
follows from the contingency-table inequality, which does not apply to a goodness-of-fit
test against fixed proportions.

| Feature | Max gate | Windows above 1 |
|---|---|---|
| RatecodeID (unseen category, S6) | 2,942.93 | 2,059 |
| RatecodeID | 1.3517 | 2,058 |
| store_and_fwd_flag | 6.4948 | 66 |
| VendorID | 1.6631 | 18 |

The gate was not changed after this was found, because changing it would invalidate
every run.

---

## 6. The eight runs

Activation at row 20,584,860, 2024-07-03 23:35:25, identical in all eight runs. Each run
produced 4,117 windows and 90,574 drift scores. In the seven runs whose console logs
survive, the producer published between 6,602 and 6,684 records per second on average
(Section 4.6 of the thesis).

| ID | Scenario | Target | Alerts |
|---|---|---|---|
| `C0_20260902T193926_7f1e3f` | clean control | none | 11,022 |
| `S1_20260904T183253_c55624` | numerical mean shift (+0.75 SD) | `fare_amount` | 13,983 |
| `S2_20260905T061550_7ca686` | categorical value swap | `payment_type` | 13,598 |
| `S3_20260905T092259_de6186` | rare-category shift (99: 0.88% to 25%) | `RatecodeID` | 14,827 |
| `S4_20260905T122634_0e865c` | variance shift (log scale x2) | `trip_distance` | 12,827 |
| `S5_20260905T153144_6ed0d7` | S1 (+0.5 SD) and S2 combined | both | 16,559 |
| `S6_20260908T193520_1db87d` | unseen category (88: 0% to 25%) | `RatecodeID` | 14,827 |
| `S7_20260919T055534_20611a` | schema change, no injection | none | 11,022 |

The Git commit of every run is listed in Appendix B of the thesis. All of these commits
are part of `main` in the published repository.

The five pilot runs are registered as well but excluded from classification: a different
slice at different window sizes has no comparable control. `classify_alerts.py` filters
them out.

### S3 correction and S6

S3 was designed as an unseen-category injection on the assumption that
`RatecodeID = 99` was absent from the reference. It is present at 0.88%. The exposé put the
unseen-category case in the protected core tier, so relabelling S3 left a core-tier
commitment unmet. S6 restores it. The injection function now reads the stored reference
and refuses a value that is already present.

### S7, added 19 September

S7 introduces `trip_source_system`, a column absent from `schema_baseline.json`, at the
activation point. Nothing else is altered. It was added to exercise criterion C7, which
no earlier run had reached.

The producer drops the key while it is null, so records before activation carry 19 fields
and records after it carry 20. Without that guard the schema change would register from
the first July window instead of from activation. Verified: no flagged window ends before
activation.

---

## 7. Results

### 7.1 Clean control

11,022 alerts from 90,574 scores = 12.17%. Range across pairs: `payment_type` with
chi-square 55.57% down to `RatecodeID` with PSI 0.02%. Six of the 22 pairs never alert in
any run. `fare_amount` with KS in the control: mean D 0.0701, maximum 0.2208, more than
double the gate.

### 7.2 Detection recall

PSI numerical 0 of 3, PSI categorical 4 of 4, KS 3 of 3, chi-square 4 of 4.
Pipeline recall 1.00 in all six scenarios with an injected target.
Every detecting metric alerted in 2,060 of 2,060 windows after activation, except
`RatecodeID` with PSI in S3 and S6 at 2,059 (the same window in both).

### 7.3 Latency

Window-close interval 2 h 54 min 10 s, identical in all eight runs. Every injected target
was detected in the first window that closed after activation. Internal check:
`total_amount` with KS produces 661 alerts after activation when it is untouched, the first
after 1 day 01:14:25; where the shift propagates into it, 2,060 of 2,060 at the floor.

### 7.4 Classification: 108,665 alerts, eight runs

| Label | Count | Share | Action(s) |
|---|---|---|---|
| BASELINE_VARIATION | 81,976 | 75.4% | SUPPRESS 75,950, INVESTIGATE 6,026 |
| CONFIRMED_DETECTION | 22,583 | 20.8% | ACT 21,541, INVESTIGATE 1,042 |
| EXPECTED_PROPAGATION | 4,106 | 3.8% | ACCEPT 4,106 |
| UNEXPLAINED | 0 | 0.0% | none |

Framework: SUPPRESS 75,950 (69.9%), ACT 21,541 (19.8%), INVESTIGATE 7,068 (6.5%),
ACCEPT 4,106 (3.8%). An operator examines 30.1% of the alerts.

The mapping from label to action is not one-to-one. That is the S7 result (Section 7.6).

### 7.5 Sensitivity of the zero

| Minimum control evidence | Unattributed | Share |
|---|---|---|
| Any single window (as implemented) | 0 | 0.00% |
| At least 1% of windows | 108 | 0.10% |
| At least 5% of windows | 1,732 | 1.59% |
| At least 10% of windows | 3,828 | 3.52% |

Report these together with the zero, not the zero alone.

### 7.6 S6, the unseen category

| Condition | Chi-square (after activation) | PSI (after activation) |
|---|---|---|
| C0 | 282 | 0.026 |
| S3 rare category | 78,194 | 1.041 |
| S6 unseen category | 394,265,150,449 | 3.501 |

Means over the 2,060 windows whose `window_end` falls at or after activation.

The chi-square figure is not a measurement. The expected count is zero, so the statistic
is undefined, and the value returned comes from the floor constant, which has no effect in
any other run. S6 and S3 produce identical alert counts and labels (14,827 / 10,721 /
4,106) despite a difference of several orders of magnitude in severity.

### 7.7 S7, the C7 result

S7 and C0 contain the same 11,022 alerts, alert for alert. 2,059 of 4,117 windows are
flagged, none before activation. C0 suppresses all 11,022; S7 sends 6,026 to INVESTIGATE.

The difference is a column that no monitored feature depends on, that the expectation suite
does not test, and that changes no value any metric reads. The flag persists because the
comparison is against a fixed baseline rather than the preceding window. Only the control
replay makes this visible.

### 7.8 Validation layer

15 timestamp failures in every run. S4 adds 131,195 on `trip_distance`.
S6 records only the 15: 706,034 records with an unseen rate code pass validation, because
the suite declares value sets for three fields and not for `RatecodeID`.

### 7.9 Without ground truth (RQ4)

C1 reads `injected_targets`, and the propagation targets of C2 are seeded from the same
list. With ground truth withheld:

| C3 rule | Recall |
|---|---|
| Binary (as implemented) | 0.0% in all scenarios |
| Percentile p95 / p99 | 100.0% |
| p99.9 / p100 | 96.0% / 92.0% |

S4 is the boundary. Its KS D of 0.1999 lies inside the control's own range
(`trip_distance` with KS, maximum 0.2596), so no rule on observable magnitude can
separate them.

C7 needs no ground truth, so the schema redirection is still seen without it.

Two threats: the percentile was chosen after the sweep, and the control covers the same
period as the scenarios, so the recall is optimistic by an amount that cannot be measured.

### 7.10 Coverage

| Element | Status |
|---|---|
| C1, C3, C4, C7 | Exercised |
| C2 | Partly: the empirical edge to ACCEPT is never reached |
| C5, C6 | Not reached; diagnostic guidance, not branch conditions |
| MONITOR | Not reached; requires an unexplained alert |

Dependency map: of the eight deterministic edges only the one from `fare_amount` was
triggered. The three empirical edges all fired, and all three resulted in SUPPRESS.

---

## 8. Principal findings

1. PSI fails silently on numerical features: 0 of 3. The mechanism is bin concentration,
   with 98.6% to 99.9995% of the reference in one bin for four of the five features,
   against 73.19% for the one that responds. Twelve bins hold exactly zero records. The
   implementation floors the empty proportions and returns about 0 instead of reporting
   the index as undefined.
2. Chi-square detects but does not communicate size. The effect size exceeds its
   theoretical maximum by three orders of magnitude, and under an unseen category the
   value reflects a constant.
3. KS does not converge. It levels off at about 17% because it measures real distance
   from a January reference. All KS tests share one 5,000-record reference sample.
4. A clean control makes the alert stream interpretable: 12.17% baseline.
5. Validation and drift detection catch different changes.
6. Latency is a property of the window size.
7. An interpretation framework that depends on ground truth is not deployable: 0.0%
   surfaced. Using the control as a distribution restores full recall at p95 and p99.
8. Alert counts discard severity: S3 and S6 are identical by count.
9. A harmless schema change turns 6,026 suppressed alerts into work (S7), because C7 is
   checked first and the flag persists.

---

## 9. Corrections made during the work

Appendix J of the thesis lists the defects found during construction and evaluation.

| Issue | Resolution |
|---|---|
| Three Kafka partitions interleaved records across activation | Single partition |
| Loading 41M rows exhausted memory | Stream month by month |
| Mean shift broke the `total_amount` identity | Propagate into `total_amount` |
| Additive variance rescaling clipped to zero | Multiplicative, log scale about the median |
| Classifier labelled pre-activation noise UNEXPLAINED | Fall through to the baseline check |
| S3 mislabelled as unseen-category | Renamed; injection reads the reference and refuses |
| Categorical gate misdescribed as Cramer's V | Redefined as normalized chi-square magnitude |
| PSI threshold attributed to a convention that omits it | Restated as a value between the bands (p. 137) |
| `classify_alerts.py` included the pilot runs | Excluded |
| Eight result CSVs had no script that wrote them | `build_result_tables.py` |
| Kafka/Parquet test read the live topic | Publishes its own fixture |
| D2 criteria C2 and C4 misdescribed the implementation | Corrected |
| Aggregated pilot PSI hid the numerical result | Split by feature type |
| S7 guard read the wrong baseline file | Reads `schema_baseline.json`; caught by a test |
| S7 dispatch nested inside the C0 branch | Fixed; all eight scenarios run through the real call path |
| Four figures inaccurate against the result data | Corrected and regenerated from the database |
| Evaluation without ground truth ignored recorded schema changes | Schema changes read from the database, so C7 can fire |
| MLflow and Airflow images lacked psycopg2, Airflow lacked curl | Added to both Dockerfiles |

---

## 10. Thesis

Arial 11 pt, 1.5 line spacing, APA 7. Page numbers below are from the table of contents.

| Part | Title | Starts on page |
|---|---|---|
| 1 | Introduction | 1 |
| 2 | Literature review and conceptual framework | 4 |
| 3 | Research methodology and evaluation design | 9 |
| 4 | Artifact architecture and implementation | 14 |
| 5 | Demonstration and experimental results | 21 |
| 6 | Alert interpretation | 30 |
| 7 | Discussion, validity and limitations | 36 |
| 8 | Conclusion and recommendations | 39 |
| | References | 41 |
| | Appendices A to P | 49 |

The chapters contain 21 numbered tables (Table 3.1 to Table 7.1, each number used once)
and five numbered figures: 2.1 (conceptual framework), 4.1 (architecture), 5.1, 5.2 and
6.1. Further tables and figures are in the appendices. The chapter text is about 14,500
words, excluding tables.

Front matter: title page, acknowledgements, abstract and keywords, contents, lists of
figures and tables. The declaration of authenticity follows the appendices.

### Disclosed divergences from the exposé

1. Persistence condition not implemented. The exposé's alert rule combines statistical
   evidence, a minimum sample size, a magnitude condition and persistence across
   consecutive windows. The fourth is absent. A three-window requirement would cut the
   control's alerts from 11,022 to 4,089 (12.17% to 4.51%) but raise the latency from
   2 h 54 min to 14 h 38 min, which would make the latency result a property of the
   persistence parameter. Disclosed in Section 4.4.
2. Thresholds not calibrated from the control. Calibration at each pair's own 99th
   percentile reduces the control from 11,022 to 923 alerts (1.02%), leaves categorical
   and KS detection intact, and costs S4 (2,060 to 1,638). Numerical PSI still exceeds a
   calibrated threshold in only 1.1% to 3.7% of windows after activation, so the PSI
   failure is not a threshold artifact. Disclosed in Sections 4.4 and 5.6.
3. Checkpoint outcomes CP1 to CP5 are reported in Section 3.7 and Appendix C.

---

## 11. References

76 entries, APA 7. Every source was checked against publisher records, DOIs or the
document itself, and the claims the thesis attributes to each source were checked as
well. 

---


## 12. Reference commands

```bash
cd ~/thesis/data-quality-pipeline && source .venv/bin/activate
docker compose up -d && sleep 25

pytest tests/ -q                       # 88 tests
python scripts/classify_alerts.py      # TRUNCATE, then rebuild all eight runs
python scripts/apply_framework.py      # D4, action per alert
python scripts/build_result_tables.py  # result CSVs from the database
python scripts/generate_figures.py     # 11 figures
python scripts/blind_mode_eval.py      # RQ4, without ground truth
python scripts/blind_mode_v2.py        # percentile sweep

# integrity check after a restart, expect 108,665
docker exec -i pipeline-postgres psql -U pipeline_admin -d postgres -c \
  "SELECT COUNT(*) FROM dq_monitoring.alert_classifications;"

# back up the database (the part that cannot be recreated quickly)
docker exec pipeline-postgres pg_dump -U pipeline_admin -d postgres -n dq_monitoring \
  | gzip > /mnt/d/dq_monitoring-$(date +%Y%m%d-%H%M).sql.gz
```

The analysis scripts must run in this order: classification first, then the framework,
then the tables and figures, because each step reads what the previous one wrote.

Running an experiment (about 3 hours: 103 to 104 min produce, 29 to 30 min validation,
37 to 47 min drift, 12 min schema, from the seven surviving run logs):

```bash
tmux new-session -d -s s1 "cd ~/thesis/data-quality-pipeline && source .venv/bin/activate && \
python scripts/run_experiment.py --scenario S1 --warmup-rows 20584860 --window-size 10000 \
--timeout-seconds 180 --report-every 2000000 2>&1 | tee logs/logs_S1.txt"

python scripts/verify_experiment.py <experiment_id>   # after each run
```

Do not truncate the database between scenarios: classification needs C0.

---

## 13. Notes for resuming

- Pause Windows Update and disable sleep before any long run. Three runs were lost or
  damaged this way, and `tee` overwrote the S6 log when the same experiment was started
  twice.
- Never start the same experiment twice.
- Docker Desktop must be running before `docker compose up`.
- Database backups are on `D:`; the latest includes S7.
- Create new files in the project from the WSL terminal rather than copying downloads in
  through Windows, which adds `Zone.Identifier` files.
