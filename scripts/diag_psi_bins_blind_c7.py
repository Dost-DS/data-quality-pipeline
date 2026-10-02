"""Read-only diagnostics for the thesis revision.

Part A  PSI and empty reference bins.
        Rebuilds the drift consumer's 10,000-record windows from the raw
        Parquet files, in the same order the producer published them, and
        counts per window how many values land in each reference bin. It then
        reports, for every numerical feature:
          - windows in which a reference-empty bin receives values
            (strict PSI undefined because the expected proportion is zero)
          - windows in which a populated reference bin receives nothing
            (strict PSI undefined because the actual proportion is zero)
          - how much of the reported PSI the floored terms contribute
          - values dropped because they fall outside the reference range
            (np.histogram ignores them)
        As a check on the reconstruction, the recomputed PSI is compared with
        the value stored in drift_scores for the same window.
        Scenarios: C0 (unmodified), S1 (mean shift), S4 (variance shift).

Part B  Blind mode with criterion C7.
        Repeats the ground-truth-withheld evaluation, but passes the schema
        flag, which an operator can observe, to the framework. Runs both the
        original C3 (binary control alert) and the magnitude-aware C3.

Writes results/diag_psi_bins.csv, results/diag_blind_c7.csv and
results/diag_blind_v2_c7.csv. Nothing in the database is modified.

Run from the repository root:  python scripts/diag_psi_bins_blind_c7.py
Options: --skip-psi, --skip-blind, --scenarios C0 S1 S4
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASELINE_PATH = "great_expectations/reference_baseline.json"
WINDOW = 10_000
WARMUP_ROWS = 20_584_860          # activation row used by every full run
EPS = 1e-6                        # same floor as kafka_consumer_drift.py
PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")

# bin codes: 0..9 reference bins, 10 = null, 11 = below range, 12 = above range
NULL, BELOW, ABOVE, NCODES = 10, 11, 12, 13


# --------------------------------------------------------------------------
# Part A
# --------------------------------------------------------------------------
def bin_codes(values, edges):
    """Bin index exactly as np.histogram assigns it; codes for excluded values."""
    v = np.asarray(values, dtype=float)
    e = np.asarray(edges, dtype=float)
    idx = np.searchsorted(e, v, side="right") - 1
    idx[v == e[-1]] = len(e) - 2                      # last bin is closed
    codes = idx.astype(np.int16)
    codes[v < e[0]] = BELOW
    codes[v > e[-1]] = ABOVE
    codes[np.isnan(v)] = NULL
    return codes


def psi_from_counts(ref_counts, cur_counts):
    """The consumer's PSI, plus the share of it contributed by floored terms."""
    b = np.asarray(ref_counts, dtype=float)
    c = np.asarray(cur_counts, dtype=float)
    if b.sum() == 0 or c.sum() == 0:
        return None, None
    bp = np.clip(b / b.sum(), EPS, None)
    cp = np.clip(c / c.sum(), EPS, None)
    terms = (cp - bp) * np.log(cp / bp)
    floored = (b == 0) | (c == 0)
    return float(terms.sum()), float(terms[floored].sum())


def window_stream(scenario, features):
    """Yield per-window code arrays {feature: codes} in publication order."""
    from run_experiment import month_paths, count_rows, load_month
    from drift_injection import apply_scenario

    paths = month_paths(None)
    counts = count_rows(paths)
    cum, act_file, act_offset = 0, None, None
    for i, n in enumerate(counts):
        if cum + n > WARMUP_ROWS:
            act_file, act_offset = i, WARMUP_ROWS - cum
            break
        cum += n

    carry = {f: np.empty(0, dtype=np.int16) for f in features}
    for i, path in enumerate(paths):
        t0 = time.time()
        df = load_month(path)
        if scenario != "C0":
            if i == act_file:
                df, _ = apply_scenario(df, scenario, act_offset)
            elif i > act_file:
                df, _ = apply_scenario(df, scenario, 0)
        codes = {f: np.concatenate([carry[f], bin_codes(df[f].values, EDGES[f])])
                 for f in features}
        del df
        n = len(codes[features[0]])
        full = (n // WINDOW) * WINDOW
        for s in range(0, full, WINDOW):
            yield {f: codes[f][s:s + WINDOW] for f in features}
        carry = {f: codes[f][full:] for f in features}
        print(f"    {scenario} {os.path.basename(path)} done "
              f"({time.time() - t0:.0f}s)", flush=True)
    if len(carry[features[0]]):
        yield carry                                    # final partial window


def stored_psi(scenario):
    """Stored numerical PSI values per feature, in window insertion order."""
    try:
        import psycopg2
    except ImportError:
        return None, None
    conn = psycopg2.connect(**PG)
    with conn.cursor() as cur:
        cur.execute("""SELECT experiment_id FROM dq_monitoring.experiments
                       WHERE scenario = %s
                         AND (ground_truth->>'total_rows')::bigint = 41169720
                       ORDER BY started_at DESC LIMIT 1""", (scenario,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return None, None
        exp = row[0]
        cur.execute("""SELECT feature_name, metric_value FROM dq_monitoring.drift_scores
                       WHERE experiment_id = %s AND metric_name = 'psi'
                         AND feature_type = 'numerical'
                       ORDER BY id""", (exp,))
        out = {}
        for f, v in cur.fetchall():
            out.setdefault(f, []).append(v)
    conn.close()
    return exp, out


def part_a(scenarios):
    global EDGES
    base = json.load(open(BASELINE_PATH))["numerical"]
    features = list(base)
    EDGES = {f: base[f]["bin_edges"] for f in features}
    ref = {f: np.asarray(base[f]["hist_counts"], dtype=float) for f in features}
    empty_ref = {f: np.where(ref[f] == 0)[0].tolist() for f in features}
    print("\n=== Part A: reference bins ===")
    for f in features:
        print(f"  {f:<16} empty reference bins: {empty_ref[f]}")

    rows = []
    for scen in scenarios:
        exp, stored = (None, None)
        try:
            exp, stored = stored_psi(scen)
        except Exception as e:                          # DB optional for Part A
            print(f"  (could not read stored PSI for {scen}: {e})")
        print(f"\n--- {scen}  stored experiment: {exp} ---", flush=True)

        acc = {f: Counter() for f in features}
        max_diff = {f: 0.0 for f in features}
        max_floored = {f: 0.0 for f in features}
        max_psi = {f: 0.0 for f in features}
        for w, win in enumerate(window_stream(scen, features)):
            for f in features:
                cnt = np.bincount(win[f], minlength=NCODES)
                cur = cnt[:10]
                a = acc[f]
                a["windows"] += 1
                a["values_null"] += int(cnt[NULL])
                a["values_below"] += int(cnt[BELOW])
                a["values_above"] += int(cnt[ABOVE])
                a["windows_out_of_range"] += int(cnt[BELOW] + cnt[ABOVE] > 0)
                into_empty = int(cur[ref[f] == 0].sum())
                a["values_in_empty_ref_bins"] += into_empty
                a["windows_mass_in_empty_ref_bin"] += int(into_empty > 0)
                a["windows_empty_window_bin"] += int(((cur == 0) & (ref[f] > 0)).any())
                strict_undefined = into_empty > 0 or ((cur == 0) & (ref[f] > 0)).any()
                a["windows_strict_psi_undefined"] += int(strict_undefined)
                v, fl = psi_from_counts(ref[f], cur)
                if v is None:
                    a["windows_psi_none"] += 1
                    continue
                max_psi[f] = max(max_psi[f], v)
                max_floored[f] = max(max_floored[f], fl)
                if stored and f in stored and w < len(stored[f]) \
                        and stored[f][w] is not None:
                    max_diff[f] = max(max_diff[f], abs(stored[f][w] - v))
                    a["windows_compared"] += 1

        for f in features:
            a = acc[f]
            r = dict(scenario=scen, feature=f,
                     empty_ref_bins=len(empty_ref[f]),
                     windows=a["windows"],
                     windows_mass_in_empty_ref_bin=a["windows_mass_in_empty_ref_bin"],
                     values_in_empty_ref_bins=a["values_in_empty_ref_bins"],
                     windows_empty_window_bin=a["windows_empty_window_bin"],
                     windows_strict_psi_undefined=a["windows_strict_psi_undefined"],
                     max_psi=round(max_psi[f], 6),
                     max_floored_contribution=round(max_floored[f], 6),
                     values_below_range=a["values_below"],
                     values_above_range=a["values_above"],
                     windows_out_of_range=a["windows_out_of_range"],
                     windows_compared_to_db=a["windows_compared"],
                     max_abs_diff_vs_db=f"{max_diff[f]:.3g}")
            rows.append(r)
            print("  " + ", ".join(f"{k}={v}" for k, v in r.items()
                                   if k not in ("scenario",)), flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/diag_psi_bins.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote results/diag_psi_bins.csv")


# --------------------------------------------------------------------------
# Part B
# --------------------------------------------------------------------------
def part_b():
    import psycopg2
    import psycopg2.extras
    from alert_classifier import classify, control_alert_set
    from interpretation_framework import recommend, PSI_STRONG, KS_STRONG

    conn = psycopg2.connect(**PG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT feature_name, metric_name, metric_value, is_drift
                       FROM dq_monitoring.drift_scores
                       WHERE experiment_id LIKE 'C0%'""")
        ctrl_rows = cur.fetchall()
        cur.execute("""SELECT experiment_id, window_start
                       FROM dq_monitoring.schema_snapshots WHERE has_schema_change""")
        schema_windows = {(r["experiment_id"], r["window_start"]) for r in cur.fetchall()}
        cur.execute("""SELECT ac.experiment_id, ac.recommended_action AS oracle,
                              d.feature_name, d.metric_name, d.metric_value,
                              d.window_start
                       FROM dq_monitoring.alert_classifications ac
                       JOIN dq_monitoring.drift_scores d ON ac.drift_score_id = d.id
                       ORDER BY ac.id""")
        alerts = cur.fetchall()
    conn.close()
    print(f"\n=== Part B: {len(alerts):,} alerts, "
          f"{len(schema_windows):,} windows with a schema change ===")

    # B1: original C3, with the schema flag passed
    control = control_alert_set(ctrl_rows)
    per = {}
    for a in alerts:
        label, _ = classify(feature=a["feature_name"], metric=a["metric_name"],
                            window_start=a["window_start"], injected=frozenset(),
                            activation_ts=None, prop_targets={},
                            control_alerts=control)
        changed = (a["experiment_id"], a["window_start"]) in schema_windows
        blind, _ = recommend(label, a["metric_name"], a["metric_value"],
                             schema_changed=changed)
        c = per.setdefault(a["experiment_id"][:2], Counter())
        c["alerts"] += 1
        c["blind_" + blind] += 1
        c["oracle_" + a["oracle"]] += 1
        if a["oracle"] == "ACT" and blind != "SUPPRESS":
            c["detections_surfaced"] += 1
    keys = ["alerts", "oracle_ACT", "detections_surfaced", "blind_SUPPRESS",
            "blind_INVESTIGATE", "blind_ACT", "blind_MONITOR", "blind_ACCEPT"]
    print("\n  B1 original C3 + C7")
    print("  scen " + "".join(f"{k:>22}" for k in keys))
    out1 = []
    for s in sorted(per):
        print(f"  {s:<5}" + "".join(f"{per[s][k]:>22,}" for k in keys))
        out1.append([s] + [per[s][k] for k in keys])
    with open("results/diag_blind_c7.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario"] + keys)
        w.writerows(out1)

    # B2: magnitude-aware C3 (as blind_mode_v2.py), with C7 in front
    ctrl = {}
    for r in ctrl_rows:
        if r["metric_value"] is not None:
            ctrl.setdefault((r["feature_name"], r["metric_name"]), []).append(
                float(r["metric_value"]))
    pcts = [95.0, 99.0, 99.9, 100.0]
    cuts = {k: {p: float(np.percentile(v, p)) for p in pcts} for k, v in ctrl.items()}

    def strong(m, v):
        return (m == "psi" and v >= PSI_STRONG) or (m == "ks" and v >= KS_STRONG)

    out2 = []
    print("\n  B2 magnitude-aware C3 + C7 (C0 excluded, as in blind_mode_v2.py)")
    for p in pcts:
        per = {}
        for a in alerts:
            s = a["experiment_id"][:2]
            if s == "C0":
                continue
            key = (a["feature_name"], a["metric_name"])
            v = float(a["metric_value"]) if a["metric_value"] is not None else 0.0
            if (a["experiment_id"], a["window_start"]) in schema_windows:
                act = "INVESTIGATE"
            elif key in cuts and v > cuts[key][p]:
                act = "ACT" if strong(a["metric_name"], v) else "INVESTIGATE"
            else:
                act = "SUPPRESS"
            c = per.setdefault(s, Counter())
            c["alerts"] += 1
            c["shown"] += act != "SUPPRESS"
            c["oracle_shown"] += a["oracle"] != "SUPPRESS"
            if a["oracle"] == "ACT":
                c["oracle_act"] += 1
                c["surfaced"] += act != "SUPPRESS"
            if a["oracle"] == "SUPPRESS" and act != "SUPPRESS":
                c["noise_leaked"] += 1
        tot = Counter()
        for s in sorted(per):
            c = per[s]
            tot.update(c)
            rec = 100 * c["surfaced"] / c["oracle_act"] if c["oracle_act"] else float("nan")
            out2.append([p, s, c["alerts"], c["oracle_act"], c["surfaced"],
                         round(rec, 2), c["noise_leaked"], c["shown"], c["oracle_shown"]])
        rec = 100 * tot["surfaced"] / tot["oracle_act"]
        print(f"  p{p:<5} recall {rec:6.2f}%  shown {tot['shown']:>7,}  "
              f"oracle shown {tot['oracle_shown']:>7,}  noise leaked "
              f"{tot['noise_leaked']:>7,}  of {tot['alerts']:,} alerts")
    with open("results/diag_blind_v2_c7.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["percentile", "scenario", "alerts", "oracle_act", "surfaced",
                    "recall_pct", "noise_leaked", "shown", "oracle_shown"])
        w.writerows(out2)
    print("\nwrote results/diag_blind_c7.csv and results/diag_blind_v2_c7.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-psi", action="store_true")
    ap.add_argument("--skip-blind", action="store_true")
    ap.add_argument("--scenarios", nargs="*", default=["C0", "S1", "S4"])
    a = ap.parse_args()
    if not a.skip_blind:
        part_b()
    if not a.skip_psi:
        part_a(a.scenarios)
