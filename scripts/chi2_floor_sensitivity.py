"""How much of the reported chi-square magnitude is the floor constant? (F8) 

Reconstructs an S3-style window from the recorded RatecodeID baseline
proportions with an unseen category at 25 percent, then sweeps the
expected-count floor. The pipeline stores only the statistic, not the
per-window category counts, so this is a reconstruction of the injection
design rather than a replay of the recorded windows. Read-only.
"""
import json
import numpy as np
from scipy import stats

BASELINE = "great_expectations/reference_baseline.json"
WINDOW_ROWS = 10_000
UNSEEN_SHARE = 0.25
FLOORS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]   # 1e-3 is the implemented value


def chi2_with_floor(baseline_counts, obs_counts, factor):
    cats = sorted(set(baseline_counts) | set(obs_counts))
    b = np.array([baseline_counts.get(c, 0) for c in cats], float)
    o = np.array([obs_counts.get(c, 0) for c in cats], float)
    expected = b / b.sum() * o.sum()
    pos = expected[expected > 0]
    floor = max(pos.min() * factor, 1e-6) if pos.size else 1e-6
    expected = np.clip(expected, floor, None)
    expected = expected / expected.sum() * o.sum()
    stat, _ = stats.chisquare(f_obs=o, f_exp=expected)
    k = len(cats)
    return float(stat), float(np.sqrt(stat / (o.sum() * max(k - 1, 1))))


def main():
    bc = json.load(open(BASELINE))["categorical"]["RatecodeID"]["value_counts"]
    total = sum(bc.values())
    print(f"RatecodeID baseline: {len(bc)} categories, {total:,} records")
    for c, n in sorted(bc.items(), key=lambda x: -x[1]):
        print(f"  {c:<6}{n:>10,}  {100*n/total:>6.2f}%")

    n_unseen = int(WINDOW_ROWS * UNSEEN_SHARE)
    obs = {c: int(round(n / total * (WINDOW_ROWS - n_unseen))) for c, n in bc.items()}
    obs["99"] = n_unseen
    print(f"\nreconstructed window: {sum(obs.values()):,} rows, "
          f"category 99 at {100*n_unseen/sum(obs.values()):.1f}%\n")

    print(f"{'floor factor':>14}{'chi2':>16}{'gate value':>14}{'vs 1e-3':>12}")
    ref = None
    for f in FLOORS:
        stat, gate = chi2_with_floor(bc, obs, f)
        if f == 1e-3:
            ref = stat
        print(f"{f:>14.0e}{stat:>16,.0f}{gate:>14.4f}"
              + (f"{stat/ref:>11.2f}x" if ref else f"{'':>12}"))

    stat_ref, _ = chi2_with_floor(bc, obs, 1e-3)
    print(f"\nrecorded S3 chi2: mean 78,262, max 109,629 (project record)")
    print(f"reconstruction at the implemented floor: {stat_ref:,.0f}")


if __name__ == "__main__":
    main()
