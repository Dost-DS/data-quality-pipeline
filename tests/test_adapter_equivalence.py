"""Validator substitution evidence.

Two assertions support the substitutability claim made in Chapter 4:

  1. each of the 32 expectations is deliberately violated at least once,
     and both implementations agree on success and unexpected_count
  2. real corpus windows produce identical outcomes in both, including
     the naturally corrupted pickup timestamps

Agreement on clean data alone would be near-vacuous, since 30 of the 32
expectations never fail anywhere in the corpus.
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from plain_validator import PlainValidator, load_suite

SUITE = load_suite()
WINDOW = 10_000
CORPUS_WINDOWS = 3

CLEAN_ROW = {
    "VendorID": 1, "tpep_pickup_datetime": pd.Timestamp("2024-06-01T10:00:00"),
    "tpep_dropoff_datetime": pd.Timestamp("2024-06-01T10:20:00"),
    "passenger_count": 1.0, "trip_distance": 3.0, "RatecodeID": 1.0,
    "store_and_fwd_flag": "N", "PULocationID": 100, "DOLocationID": 200,
    "payment_type": 1, "fare_amount": 15.0, "extra": 0.0, "mta_tax": 0.5,
    "tip_amount": 2.0, "tolls_amount": 0.0, "improvement_surcharge": 0.3,
    "total_amount": 17.8, "congestion_surcharge": 2.5, "Airport_fee": 0.0,
}


@pytest.fixture(scope="session")
def gx_suite():
    gx = pytest.importorskip("great_expectations")
    ctx = gx.get_context(mode="file", project_root_dir="great_expectations")
    try:
        ds = ctx.data_sources.get("taxi_datasource")
    except Exception:
        ds = ctx.data_sources.add_pandas(name="taxi_datasource")
    try:
        asset = ds.get_asset("equiv_asset")
    except Exception:
        asset = ds.add_dataframe_asset(name="equiv_asset")
    try:
        bd = asset.get_batch_definition("equiv_batch")
    except Exception:
        bd = asset.add_batch_definition_whole_dataframe("equiv_batch")
    return bd, ctx.suites.get("taxi_trip_suite")


@pytest.fixture(scope="session")
def plain():
    return PlainValidator(SUITE)


def gx_outcomes(gx_suite, df) -> dict:
    bd, suite = gx_suite
    res = bd.get_batch(batch_parameters={"dataframe": df}).validate(
        suite, result_format="COMPLETE")
    out = {}
    for r in res.results:
        cfg = r.expectation_config
        out[(cfg.type, cfg.kwargs.get("column"))] = (
            bool(r.success), int(r.result.get("unexpected_count") or 0))
    return out


def plain_outcomes(plain, df) -> dict:
    res = plain.validate(df)
    return {(r["expectation"], r["column"]):
            (bool(r["success"]), int(r["unexpected_count"]))
            for r in res["results"]}


def diff(a: dict, b: dict) -> list[str]:
    keys = sorted(set(a) | set(b), key=lambda k: (k[0], str(k[1])))
    return [f"{k[0]} on {k[1]}: gx={a.get(k)} plain={b.get(k)}"
            for k in keys if a.get(k) != b.get(k)]


def clean_frame(n=200) -> pd.DataFrame:
    return pd.DataFrame([CLEAN_ROW] * n)


def violate(df, etype, col, kw) -> pd.DataFrame:
    """Return a copy breaching exactly one expectation."""
    d = df.copy()
    if etype == "expect_column_to_exist":
        return d.drop(columns=[col])
    n = len(d)
    mostly = kw.get("mostly") or 1.0
    k = min(n, max(1, int(n * (1 - mostly)) + 2))
    idx = d.index[:k]
    if etype == "expect_column_values_to_not_be_null":
        d.loc[idx, col] = None
    elif etype == "expect_column_values_to_be_between":
        hi = kw.get("max_value")
        d[col] = d[col].astype(object)
        d.loc[idx, col] = (pd.Timestamp("2030-01-01") if isinstance(hi, str)
                           else float(hi) * 10 + 1000)
    elif etype == "expect_column_values_to_be_in_set":
        vs = kw["value_set"]
        d[col] = d[col].astype(object)
        d.loc[idx, col] = "ZZ" if isinstance(vs[0], str) else 999
    else:
        raise ValueError(etype)
    return d


IDS = [f"{(e.get('expectation_type') or e.get('type')).replace('expect_column_','')}"
       f"__{e['kwargs'].get('column')}" for e in SUITE]


def test_clean_window_agrees(gx_suite, plain):
    df = clean_frame()
    d = diff(gx_outcomes(gx_suite, df), plain_outcomes(plain, df))
    assert not d, "clean window disagreement:\n  " + "\n  ".join(d)


@pytest.mark.parametrize("exp", SUITE, ids=IDS)
def test_each_expectation_violated_agrees(gx_suite, plain, exp):
    etype = exp.get("expectation_type") or exp.get("type")
    kw = exp.get("kwargs", {})
    col = kw.get("column")
    df = violate(clean_frame(), etype, col, kw)

    g, p = gx_outcomes(gx_suite, df), plain_outcomes(plain, df)
    assert g.get((etype, col), (True, 0))[0] is False, \
        f"the constructed violation did not fail {etype} on {col} in GX"
    d = diff(g, p)
    assert not d, f"disagreement after violating {etype} on {col}:\n  " + "\n  ".join(d)


def test_corpus_windows_agree(gx_suite, plain):
    files = sorted(glob.glob("data/raw/*.parquet"))
    if not files:
        pytest.skip("data/raw parquet files not present")
    df = pd.read_parquet(files[0]).head(WINDOW * CORPUS_WINDOWS)
    for i in range(CORPUS_WINDOWS):
        w = df.iloc[i * WINDOW:(i + 1) * WINDOW]
        if w.empty:
            break
        d = diff(gx_outcomes(gx_suite, w), plain_outcomes(plain, w))
        assert not d, f"corpus window {i} disagreement:\n  " + "\n  ".join(d)


def test_corrupted_timestamps_agree(gx_suite, plain):
    """The 15 naturally corrupted pickup timestamps (project record, sec. 3)."""
    files = sorted(glob.glob("data/raw/*.parquet"))
    if not files:
        pytest.skip("data/raw parquet files not present")
    df = pd.read_parquet(files[0])
    bad = df[df["tpep_pickup_datetime"] < pd.Timestamp("2024-01-01")]
    if bad.empty:
        pytest.skip("no corrupted pickup timestamps in this month")
    w = pd.concat([bad, df.head(500)])
    g, p = gx_outcomes(gx_suite, w), plain_outcomes(plain, w)
    key = ("expect_column_values_to_be_between", "tpep_pickup_datetime")
    assert g[key] == p[key], f"gx={g[key]} plain={p[key]}"
    d = diff(g, p)
    assert not d, "disagreement on corrupted rows:\n  " + "\n  ".join(d)
