"""Compute and persist the reference baselines from the first week of January 2024. 

Produces two artefacts:
  reference_baseline.json  bin edges and counts for numerical features,
                           value frequencies for categorical features,
                           and a fixed random sample of raw values for the KS test
  schema_baseline.json     column names and coarse types
"""
import json
import os

import numpy as np
import pandas as pd

DATA_PATH = "data/raw/yellow_tripdata_2024-01.parquet"
OUT_DIR = "great_expectations"
NUM_BINS = 10
KS_SAMPLE = 5000
SEED = 42

NUMERICAL = ["trip_distance", "fare_amount", "total_amount",
             "passenger_count", "tip_amount"]
CATEGORICAL = ["VendorID", "payment_type", "RatecodeID",
               "store_and_fwd_flag", "PULocationID", "DOLocationID"]

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2024-01-08")


def coarse_type(dtype):
    d = str(dtype)
    if "int" in d:
        return "integer"
    if "float" in d:
        return "float"
    if "datetime" in d:
        return "datetime"
    if "bool" in d:
        return "boolean"
    return "string"


df = pd.read_parquet(DATA_PATH)
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
week = df[(df["tpep_pickup_datetime"] >= START) &
          (df["tpep_pickup_datetime"] < END)].copy()

print(f"baseline window: {START.date()} to {END.date()}")
print(f"baseline rows:   {len(week):,}")

baseline = {
    "window_start": START.isoformat(),
    "window_end": END.isoformat(),
    "row_count": int(len(week)),
    "num_bins": NUM_BINS,
    "ks_sample_size": KS_SAMPLE,
    "seed": SEED,
    "numerical": {},
    "categorical": {},
}

for col in NUMERICAL:
    s = week[col].dropna()
    if s.empty:
        continue
    edges = np.linspace(s.min(), s.max(), NUM_BINS + 1)
    counts, _ = np.histogram(s, bins=edges)
    baseline["numerical"][col] = {
        "bin_edges": edges.tolist(),
        "hist_counts": counts.tolist(),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "raw_values_sample": s.sample(min(KS_SAMPLE, len(s)),
                                      random_state=SEED).tolist(),
    }

for col in CATEGORICAL:
    s = week[col].dropna().astype(str)
    if s.empty:
        continue
    baseline["categorical"][col] = {
        "value_counts": s.value_counts().to_dict(),
        "n_categories": int(s.nunique()),
    }

with open(os.path.join(OUT_DIR, "reference_baseline.json"), "w") as f:
    json.dump(baseline, f, indent=2)

schema = {c: coarse_type(d) for c, d in df.dtypes.items()}
with open(os.path.join(OUT_DIR, "schema_baseline.json"), "w") as f:
    json.dump(schema, f, indent=2)

print(f"\nnumerical features:   {list(baseline['numerical'])}")
print(f"categorical features: {list(baseline['categorical'])}")
for col, meta in baseline["categorical"].items():
    print(f"  {col:<20} {meta['n_categories']:>4} categories")
print(f"\nschema baseline: {len(schema)} columns")
