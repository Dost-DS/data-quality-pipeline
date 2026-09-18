"""Generate the figures for the results chapters.

Reads directly from PostgreSQL so that every figure is reproducible from the
stored experiment data rather than from intermediate files.

Two figures were corrected after the alert interpretation framework was
verified against its implementation:

  fig_window_pilot   PSI is now split by feature type. The aggregated series
                     conflated numerical PSI, which never alerts, with
                     categorical PSI, which does, and so presented the metric
                     as the quietest of the three when the numerical half is
                     simply silent.

  fig_decision_tree  Rebuilt from interpretation_framework.recommend() and
                     alert_classifier.classify() rather than from the design
                     description. The previous version omitted the empirical
                     dependency edge entirely and routed chi-square through a
                     magnitude test the implementation skips.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import psycopg2

PG = dict(host=os.environ.get("PG_HOST", "localhost"), port=5432,
          dbname="postgres", user="pipeline_admin", password="pipeline_pass")
OUT = "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BLUE, ORANGE, GREEN, RED, GREY = "#2E5C8A", "#E08E45", "#2E7D46", "#B23A48", "#888888"
PALE = "#A8BBD0"          # unexercised decision-tree nodes


def q(sql):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


# ---------------------------------------------------------------- figure 1
def fig_window_pilot():
    """Baseline alert rate against window size, per metric and feature type.

    Not a false-alert rate: where the data has genuinely moved away from the
    January reference, an alert is a correct detection of a change nobody
    asked about. See Chapter 3, Section 3.5.
    """
    df = q("""
        SELECT (e.ground_truth->>'window_size')::int AS window_size,
               d.metric_name, d.feature_type,
               COUNT(*)                           AS scores,
               COUNT(*) FILTER (WHERE d.is_drift) AS alerts
        FROM dq_monitoring.drift_scores d
        JOIN dq_monitoring.experiments e USING (experiment_id)
        WHERE e.scenario = 'PILOT'
        GROUP BY 1, 2, 3""")

    split = not df.empty
    if not split:
        print("  ! no PILOT rows in drift_scores - falling back to the CSV, which")
        print("    cannot separate numerical from categorical PSI. Re-run")
        print("    scripts/pilot_window_size.py to recover the breakdown.")
        df = pd.read_csv("results/pilot_window_size.csv").rename(
            columns={"false_alerts": "alerts"})
        df["feature_type"] = "all"

    df["pct"] = 100.0 * df.alerts / df.scores

    # KS is computed on numerical features only and chi-square on categorical
    # only, so neither needs splitting. Only PSI is applied to both.
    if split:
        series = [("psi",  "numerical",   "PSI (numerical)",   BLUE,   "o", "--", "white"),
                  ("psi",  "categorical", "PSI (categorical)", BLUE,   "o", "-",  BLUE),
                  ("ks",   "numerical",   "KS (numerical)",    ORANGE, "s", "-",  ORANGE),
                  ("chi2", "categorical", "Chi\u00b2 (categorical)", GREEN, "^", "-", GREEN)]
    else:
        series = [("psi",  "all", "PSI (both feature types)", BLUE,   "o", "-", BLUE),
                  ("ks",   "all", "KS (numerical)",           ORANGE, "s", "-", ORANGE),
                  ("chi2", "all", "Chi\u00b2 (categorical)", GREEN,  "^", "-", GREEN)]

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    plotted = {}
    for metric, ftype, label, colour, marker, style, mfc in series:
        d = df[(df.metric_name == metric) & (df.feature_type == ftype)]
        if d.empty:
            continue
        d = d.sort_values("window_size")
        ax.plot(d.window_size, d.pct, marker=marker, color=colour, label=label,
                linewidth=2, markersize=7, linestyle=style,
                markerfacecolor=mfc, markeredgewidth=1.6, zorder=3)
        plotted[label] = [round(x, 2) for x in d.pct]

    ax.set_xscale("log")
    ax.set_xticks(sorted(df.window_size.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Window size (records)")
    ax.set_ylabel("Baseline alert rate (%)")
    ax.set_title("Baseline alert rate by window size on unmodified data")
    ax.set_ylim(bottom=-1.2)
    ax.axvline(10000, color=GREY, linestyle="--", linewidth=1, zorder=1)
    ax.text(10400, ax.get_ylim()[1] * 0.94, "adopted", fontsize=9, color=GREY)
    ax.legend(frameon=False, fontsize=9, loc="center left")

    note = None
    npsi = plotted.get("PSI (numerical)")
    if npsi is not None and max(npsi) == 0:
        note = "Numerical PSI raised no alert at any window size."
    elif not split:
        note = "PSI is aggregated over both feature types; see console warning."
    if note:
        ax.text(0.985, 0.05, note, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8.5, style="italic", color="#555555")

    fig.savefig(f"{OUT}/fig_window_pilot.png")
    plt.close(fig)
    print("fig_window_pilot.png" + ("" if split else "   [AGGREGATED - see warning]"))
    for k, v in plotted.items():
        print(f"    {k:<26}" + "  ".join(f"{x:6.2f}" for x in v))


# ---------------------------------------------------------------- figure 2
def fig_recall():
    """Detection outcome per injected target and metric. Unchanged."""
    df = pd.read_csv("results/recall_by_metric.csv")
    df["detected"] = df.post_alerts > 0
    df["row"] = df.scenario + "  " + df.injected_feature
    pivot = df.pivot_table(index="row", columns="metric_name",
                           values="detected", aggfunc="first")
    pivot = pivot[[c for c in ["psi", "ks", "chi2"] if c in pivot.columns]]

    n_r, n_c = pivot.shape
    fig, ax = plt.subplots(figsize=(6.5, 0.55 * n_r + 1.8))
    for i in range(n_r):
        for j in range(n_c):
            v = pivot.iloc[i, j]
            if pd.isna(v):
                face, txt, tc = "#ECECEC", "n/a", "#999999"
            elif v:
                face, txt, tc = GREEN, "detected", "white"
            else:
                face, txt, tc = RED, "missed", "white"
            ax.add_patch(plt.Rectangle((j, n_r - i - 1), 1, 1,
                                       facecolor=face, edgecolor="white", linewidth=2))
            ax.text(j + .5, n_r - i - .5, txt, ha="center", va="center",
                    fontsize=9, color=tc, fontweight="bold")
    ax.set_xlim(0, n_c); ax.set_ylim(0, n_r)
    ax.set_xticks([j + .5 for j in range(n_c)])
    ax.set_xticklabels([c.upper() for c in pivot.columns], fontweight="bold")
    ax.set_yticks([n_r - i - .5 for i in range(n_r)])
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Detection outcome by injected target and metric", pad=14)
    ax.grid(False); ax.set_frame_on(False); ax.tick_params(length=0)
    fig.savefig(f"{OUT}/fig_recall.png")
    plt.close(fig)
    print("fig_recall.png")


# ---------------------------------------------------------------- figure 3
def fig_magnitude_ratio():
    """Post-activation magnitude against the control's pre-activation mean.

    The denominator is taken from the clean control replay rather than from
    the scenario's own pre-activation windows. Those two differ slightly,
    because a small number of windows carry corrupted pickup timestamps that
    place window_start outside 2024 and are therefore counted as
    pre-activation while containing post-activation records. The control is
    the cleaner reference, and using it is stated on the axis.
    """
    act = q("""SELECT MIN(activation_timestamp) AS ts
               FROM dq_monitoring.experiments
               WHERE activation_timestamp IS NOT NULL""").ts.iloc[0]

    ctrl = q(f"""
        SELECT d.feature_name, d.metric_name,
               AVG(d.metric_value) AS control_pre_mean
        FROM dq_monitoring.drift_scores d
        JOIN dq_monitoring.experiments e USING (experiment_id)
        WHERE e.scenario = 'C0'
          AND d.window_start < TIMESTAMPTZ '{act}'
          AND d.metric_value IS NOT NULL
        GROUP BY 1, 2""")

    df = pd.read_csv("results/recall_by_metric.csv")
    # S5 is additive with S1 and S2 and would duplicate their bars. S6 is
    # excluded because its chi-square ratio exceeds 10^9: the statistic is
    # undefined under an unseen category and the value returned depends on
    # the regularisation floor, so plotting it on a shared axis would invite
    # a comparison the caption exists to forbid. S6 has its own figure.
    df = df[~df.scenario.isin(["S5", "S6"])].copy()
    df = df.merge(ctrl, left_on=["injected_feature", "metric_name"],
                  right_on=["feature_name", "metric_name"], how="left")

    disagree = (df.control_pre_mean - df.mean_pre).abs() / df.mean_pre.abs()
    if (disagree > 0.01).any():
        print("    note: stored mean_pre differs from the control pre-activation mean")
        for _, r in df[disagree > 0.01].iterrows():
            print(f"      {r.scenario} {r.injected_feature}/{r.metric_name}: "
                  f"stored {r.mean_pre:,.4f}  control {r.control_pre_mean:,.4f}")

    df["ratio"] = df.mean_post / df.control_pre_mean
    df["label"] = df.scenario + " " + df.injected_feature + "\n" + df.metric_name.str.upper()
    df = df.sort_values("ratio")

    colours = [GREEN if d else RED for d in (df.post_alerts > 0)]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.barh(df.label, df.ratio, color=colours, edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(1, color=GREY, linewidth=1)
    ax.set_xlabel("Post-activation mean \u00f7 control pre-activation mean (log scale)")
    ax.set_title("Magnitude change at the injected target")
    for y, (r, det) in enumerate(zip(df.ratio, df.post_alerts > 0)):
        ax.text(r * 1.15, y, f"{r:,.1f}\u00d7" + ("" if det else "  (no alert)"),
                va="center", fontsize=8.5)
    ax.set_xlim(0.7, df.ratio.max() * 12)
    ax.text(0.995, 0.03,
            "Ratios are not comparable across metrics: KS D is bounded by 1,\n"
            "the chi-square statistic is unbounded. See Chapter 5.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic", color="#555555")
    fig.savefig(f"{OUT}/fig_magnitude_ratio.png")
    plt.close(fig)
    print("fig_magnitude_ratio.png")


# ---------------------------------------------------------------- figure 4
def fig_classification():
    """Alert classification per scenario."""
    df = q("""SELECT e.scenario, ac.label, COUNT(*) AS n
              FROM dq_monitoring.alert_classifications ac
              JOIN dq_monitoring.experiments e USING (experiment_id)
              GROUP BY 1,2""")
    order = ["BASELINE_VARIATION", "EXPECTED_PROPAGATION",
             "CONFIRMED_DETECTION", "UNEXPLAINED"]
    colours = {"BASELINE_VARIATION": GREY, "EXPECTED_PROPAGATION": ORANGE,
               "CONFIRMED_DETECTION": GREEN, "UNEXPLAINED": RED}
    pivot = df.pivot(index="scenario", columns="label", values="n").fillna(0)
    for c in order:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot = pivot[order].sort_index()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = np.zeros(len(pivot))
    for c in order:
        ax.bar(pivot.index, pivot[c], bottom=bottom, label=c.replace("_", " ").title(),
               color=colours[c], edgecolor="white", width=.62)
        bottom += pivot[c].values
    ax.set_ylabel("Alerts")
    ax.set_title("Alert classification by scenario")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    ax.text(0.995, 0.02,
            "C0 is baseline variation by construction: the control defines the reference.\n"
            "No alert was left unattributed; Chapter 5 reports how sensitive that is to the rule.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic", color="#555555")
    fig.savefig(f"{OUT}/fig_classification.png")
    plt.close(fig)
    print("fig_classification.png")


# ---------------------------------------------------------------- figure 5
def fig_decision_tree():
    """Deliverable D1, drawn from the implementation.

    classify() then recommend():
      schema change                        -> INVESTIGATE
      injected target, after activation
          metric is chi-square             -> ACT   (magnitude not assessable)
          magnitude >= 2x threshold        -> ACT
          otherwise                        -> INVESTIGATE
      declared edge from an injected feature, after activation
          deterministic edge               -> ACCEPT
          empirical, absent from control   -> ACCEPT
          empirical, present in control    -> SUPPRESS
      same feature and metric in control   -> SUPPRESS
      otherwise (unexplained)
          magnitude >= 2x threshold        -> INVESTIGATE
          otherwise                        -> MONITOR
    """
    try:
        ex = q("""SELECT
              (SELECT COUNT(*) FROM dq_monitoring.schema_snapshots
               WHERE has_schema_change)     AS schema_changes,
              (SELECT COUNT(*) FROM dq_monitoring.alert_classifications
               WHERE label = 'UNEXPLAINED') AS unexplained""")
        saw_schema = int(ex.schema_changes.iloc[0]) > 0
        saw_unexpl = int(ex.unexplained.iloc[0]) > 0
    except Exception as exc:
        print(f"    ! could not query exercised branches ({type(exc).__name__})")
        saw_schema = saw_unexpl = True

    fig, ax = plt.subplots(figsize=(15.5, 11))
    ax.set_xlim(0, 17.4); ax.set_ylim(0, 16.6); ax.axis("off")

    def box(x, y, w, h, text, colour, fs=10, live=True):
        ax.add_patch(FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor=colour if live else PALE,
            edgecolor="white" if live else "#5A5A5A",
            linestyle="-" if live else (0, (3, 2)),
            linewidth=1.6, zorder=2))
        ax.text(x, y, text, ha="center", va="center",
                color="white" if live else "#242424",
                fontsize=fs, fontweight="bold", zorder=3)

    def arrow(x1, y1, x2, y2, lbl=None, lx=None, ly=None, live=True):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
            color="#555555" if live else "#9A9A9A", linewidth=1.3,
            linestyle="-" if live else (0, (3, 2)), zorder=1))
        if lbl:
            ax.text(lx if lx is not None else (x1 + x2) / 2,
                    ly if ly is not None else (y1 + y2) / 2 + .28,
                    lbl, fontsize=9, style="italic", ha="center", va="center",
                    color="#333333" if live else "#8A8A8A", zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.4))

    SX, SW = 3.0, 4.2

    box(SX, 15.6, SW, .62, "Alert on feature X", BLUE)
    box(SX, 14.3, SW, .95, "C7  Schema change\nin this window?", BLUE)
    arrow(SX, 15.29, SX, 14.78)
    box(8.6, 14.3, 2.9, .62, "INVESTIGATE", GREEN, 11, live=saw_schema)
    arrow(5.1, 14.3, 7.15, 14.3, "yes", 6.1, 14.62, live=saw_schema)

    # C1 -------------------------------------------------------------------
    box(SX, 12.4, SW, 1.05, "C1  Injected target,\nwindow at/after activation?", BLUE, 9.5)
    arrow(SX, 13.82, SX, 12.93, "no", SX + .42, 13.38)
    box(7.9, 12.4, 2.9, .95, "Metric is\nchi-square?", BLUE)
    arrow(5.1, 12.4, 6.45, 12.4, "yes", 5.8, 12.72)
    box(15.5, 12.9, 2.1, .62, "ACT", GREEN, 11.5)
    box(11.9, 11.5, 3.4, .95, "C4  Magnitude\n\u2265 2\u00d7 threshold?", BLUE, 9.5)
    box(15.4, 10.6, 3.0, .62, "INVESTIGATE", GREEN, 10.5)
    arrow(9.35, 12.62, 14.45, 12.9, "yes", 11.6, 13.05)
    arrow(9.35, 12.1, 10.2, 11.75, "no", 9.75, 11.72)
    arrow(13.6, 11.72, 14.45, 12.68, "yes", 13.7, 12.32)
    arrow(13.6, 11.22, 13.9, 10.8, "no", 13.55, 10.86)

    # C2 -------------------------------------------------------------------
    box(SX, 9.4, SW, 1.05, "C2  Declared dependency edge\nfrom an injected feature?", BLUE, 9.5)
    arrow(SX, 11.87, SX, 9.93, "no", SX + .42, 10.9)
    box(7.9, 9.4, 2.9, .62, "Edge type?", BLUE)
    arrow(5.1, 9.4, 6.45, 9.4, "yes", 5.8, 9.72)
    box(15.5, 8.5, 2.3, .62, "ACCEPT", GREEN, 11.5)
    box(11.9, 7.4, 3.4, .95, "C3  Alerted in the\nclean control?", BLUE, 9.5)
    box(15.4, 6.1, 2.6, .62, "SUPPRESS", GREEN, 11)
    arrow(9.35, 9.25, 14.35, 8.62, "deterministic", 11.7, 9.22)
    arrow(9.35, 9.05, 10.2, 7.7, "empirical", 9.55, 8.28)
    arrow(13.6, 7.62, 14.35, 8.38, "no", 13.7, 8.16)
    arrow(13.6, 7.12, 14.1, 6.3, "yes", 13.62, 6.66)

    # C3 -------------------------------------------------------------------
    box(SX, 6.5, SW, 1.05, "C3  Same feature and metric\nalerted in the clean control?", BLUE, 9.5)
    arrow(SX, 8.87, SX, 7.03, "no", SX + .42, 7.95)
    box(8.6, 6.5, 2.6, .62, "SUPPRESS", GREEN, 11)
    arrow(5.1, 6.5, 7.3, 6.5, "yes", 6.2, 6.82)

    # unexplained ----------------------------------------------------------
    box(SX, 4.0, SW, .95, "C4  Magnitude\n\u2265 2\u00d7 threshold?", BLUE, 9.5, live=saw_unexpl)
    arrow(SX, 5.97, SX, 4.48, "no \u2014 unexplained", SX + 1.35, 5.24, live=saw_unexpl)
    box(8.6, 4.5, 2.9, .62, "INVESTIGATE", GREEN, 10.5, live=saw_unexpl)
    box(8.6, 2.9, 2.6, .62, "MONITOR", GREEN, 11, live=saw_unexpl)
    arrow(5.1, 4.18, 7.15, 4.5, "yes", 6.05, 4.72, live=saw_unexpl)
    arrow(5.1, 3.8, 7.3, 3.0, "no", 6.1, 3.14, live=saw_unexpl)

    handles = [mpatches.Patch(color=BLUE, label="Decision criterion"),
               mpatches.Patch(color=GREEN, label="Recommended action")]
    if not (saw_schema and saw_unexpl):
        handles.append(Line2D([0], [0], color="#5A5A5A", linestyle=(0, (3, 2)),
                              linewidth=1.6,
                              label="Path not exercised by the experiments"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=10.5,
              bbox_to_anchor=(0.005, 0.005))
    ax.text(0.68, 0.02,
            "Chi-square alerts bypass C4: the statistic has no bounded scale, so an\n"
            "unexplained chi-square alert always reaches MONITOR.",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, style="italic", color="#555555")
    ax.set_title("Alert Interpretation Framework: decision tree (D1)",
                 fontsize=15, fontweight="bold", pad=14)
    fig.savefig(f"{OUT}/fig_decision_tree.png")
    plt.close(fig)
    print("fig_decision_tree.png")
    print(f"    schema branch exercised: {saw_schema}; "
          f"unexplained branch exercised: {saw_unexpl}")



# ---------------------------------------------------------------- figure 6
def fig_psi_binning():
    """Maximum PSI per numerical feature against window size (pilot runs).

    The mechanism behind the numerical PSI result. Bin edges are drawn over
    the baseline's full observed range, so a feature whose range is set by an
    extreme outlier gets bins far wider than its own variability, and no
    plausible shift moves mass across a boundary. passenger_count is the only
    numerical feature whose bin width is smaller than its standard deviation,
    and the only one on which PSI responds at all.
    """
    df = q("""
        SELECT (e.ground_truth->>'window_size')::int AS ws,
               d.feature_name,
               MAX(d.metric_value) AS max_psi
        FROM dq_monitoring.drift_scores d
        JOIN dq_monitoring.experiments e USING (experiment_id)
        WHERE e.scenario = 'PILOT' AND d.metric_name = 'psi'
          AND d.feature_type = 'numerical'
        GROUP BY 1, 2""")
    if df.empty:
        print("  ! no PILOT numerical PSI rows; skipping fig_psi_binning")
        return

    # Share of the reference falling in the single most populated bin. This
    # orders the features exactly inversely to their observed maximum PSI,
    # which the bin-width ratio does not, and it states the mechanism more
    # directly: PSI sums the change in each bin's share, so where nearly the
    # whole reference sits in one bin there is nowhere for mass to move.
    ratios = {}
    try:
        import json
        b = json.load(open("great_expectations/reference_baseline.json"))
        for f, r in b["numerical"].items():
            c = r["hist_counts"]
            ratios[f] = 100.0 * max(c) / sum(c)
    except Exception as exc:
        print(f"    (modal-bin shares unavailable: {type(exc).__name__})")

    order = sorted(df.feature_name.unique(),
                   key=lambda f: -df[df.feature_name == f].max_psi.max())
    palette = [BLUE, ORANGE, GREEN, RED, "#7B5EA7"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for i, feat in enumerate(order):
        d = df[df.feature_name == feat].sort_values("ws")
        lbl = feat
        if feat in ratios:
            # four decimals: trip_distance is 99.9995% and rounds to a
            # misleading 100.00% at two, which would also contradict Table 5.6
            lbl += f"  ({ratios[feat]:.4f}% of reference in one bin)"
        ax.plot(d.ws, d.max_psi.clip(lower=1e-5), marker=markers[i % 5],
                color=palette[i % 5], label=lbl, linewidth=2, markersize=6.5)

    ax.axhline(0.2, color=GREY, linestyle="--", linewidth=1.2)
    ax.text(3200, 0.135, "alerting threshold (0.2)", fontsize=8.5, color=GREY)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sorted(df.ws.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Window size (records)")
    ax.set_ylabel("Maximum PSI observed (log scale)")
    ax.set_title("Maximum PSI per numerical feature, unmodified data")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.savefig(f"{OUT}/fig_psi_binning.png")
    plt.close(fig)
    print("fig_psi_binning.png")
    for feat in order:
        d = df[df.feature_name == feat].sort_values("ws")
        r = f"  modal share {ratios[feat]:>9.4f}%" if feat in ratios else ""
        print(f"    {feat:<18}" + "  ".join(f"{v:9.4f}" for v in d.max_psi) + r)


# ---------------------------------------------------------------- figure 7
def fig_unseen_category():
    """RatecodeID under three conditions: control, rare-category shift, unseen.

    Both metrics detect the unseen category. Only one reports it on a scale
    that can be reasoned about: PSI moves to roughly seventeen times its
    threshold, while the chi-square statistic reaches the order of 10^11
    because an unseen category drives the expected count to zero and the
    value returned is a function of the regularisation floor.
    """
    df = q("""
        SELECT LEFT(experiment_id, 2) AS scen, metric_name,
               AVG(metric_value) FILTER (
                   WHERE window_start >= (SELECT MIN(activation_timestamp)
                                          FROM dq_monitoring.experiments
                                          WHERE activation_timestamp IS NOT NULL)
               ) AS post
        FROM dq_monitoring.drift_scores
        WHERE feature_name = 'RatecodeID'
          AND experiment_id LIKE ANY (ARRAY['C0%', 'S3%', 'S6%'])
        GROUP BY 1, 2""")
    if df.empty or "S6" not in set(df.scen):
        print("  ! S6 not present; skipping fig_unseen_category")
        return

    labels = {"C0": "C0\ncontrol",
              "S3": "S3\nrare category\n0.88% \u2192 25%",
              "S6": "S6\nunseen category\n0% \u2192 25%"}
    order = ["C0", "S3", "S6"]
    colours = [GREY, ORANGE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    for ax, metric, title, thr in [
            (axes[0], "chi2", "Chi-square statistic", None),
            (axes[1], "psi", "Population Stability Index", 0.2)]:
        vals = [float(df[(df.scen == s) & (df.metric_name == metric)].post.iloc[0])
                for s in order]
        ax.bar([labels[s] for s in order], vals, color=colours,
               edgecolor="white", width=.6)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Post-activation mean (log scale)")
        top = max(vals)
        ax.set_ylim(min(vals) / 6, top * 40)
        for i, v in enumerate(vals):
            txt = f"{v:,.0f}" if v >= 100 else f"{v:.3f}"
            ax.text(i, v * 1.5, txt, ha="center", fontsize=9, fontweight="bold")
        if thr is not None:
            ax.axhline(thr, color=GREY, linestyle="--", linewidth=1.2)
            ax.text(-0.42, thr * 1.25, "threshold", fontsize=8, color=GREY,
                    ha="left")
    fig.suptitle("RatecodeID under three conditions", fontsize=13,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_unseen_category.png")
    plt.close(fig)
    print("fig_unseen_category.png")
    for _, r in df.sort_values(["metric_name", "scen"]).iterrows():
        print(f"    {r.scen}  {r.metric_name:<5}  post = {float(r.post):,.4f}")


# ---------------------------------------------------------------- figure 8
def fig_metric_timeline():
    """KS on fare_amount across the corpus: control against S1.

    The only figure showing the temporal dimension directly. The control
    series wanders because the reference is a single week in January and the
    corpus spans a year; the S1 series separates from it at the activation
    point and does not return.
    """
    act = q("""SELECT MIN(activation_timestamp) AS ts
               FROM dq_monitoring.experiments
               WHERE activation_timestamp IS NOT NULL""").ts.iloc[0]

    df = q(f"""
        SELECT LEFT(experiment_id, 2) AS scen, window_start, metric_value
        FROM dq_monitoring.drift_scores
        WHERE feature_name = 'fare_amount' AND metric_name = 'ks'
          AND experiment_id LIKE ANY (ARRAY['C0%', 'S1%'])
          AND window_start >= TIMESTAMPTZ '2024-01-01'
          AND window_start <  TIMESTAMPTZ '2025-01-01'
        ORDER BY window_start""")
    if df.empty:
        print("  ! no fare_amount KS rows; skipping fig_metric_timeline")
        return

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for scen, colour, lbl in [("C0", GREY, "C0  clean control"),
                              ("S1", RED, "S1  mean shift injected")]:
        d = df[df.scen == scen].sort_values("window_start")
        if d.empty:
            continue
        ax.plot(d.window_start, d.metric_value, color=colour, linewidth=.5,
                alpha=.35, zorder=2)
        ax.plot(d.window_start, d.metric_value.rolling(50, center=True).mean(),
                color=colour, linewidth=2.0, label=lbl, zorder=3)

    ax.axvline(pd.Timestamp(act), color=BLUE, linestyle="--", linewidth=1.4,
               zorder=4)
    ax.text(pd.Timestamp(act), ax.get_ylim()[1] * .97, "  activation",
            color=BLUE, fontsize=9, va="top")
    ax.axhline(0.1, color=GREEN, linestyle=":", linewidth=1.4, zorder=1)
    ax.text(pd.Timestamp("2024-01-10"), 0.108, "effect-size gate (D = 0.1)",
            color=GREEN, fontsize=8.5)

    ax.set_ylabel("Kolmogorov-Smirnov D")
    ax.set_xlabel("Window start (event time)")
    ax.set_title("KS on fare_amount across the corpus, control against S1")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    n_excl = 4117 - len(df[df.scen == "C0"])
    ax.text(0.995, 0.62,
            f"Thin lines are per-window values; bold lines a 50-window rolling mean.\n"
            f"{n_excl} windows per run are omitted: their bounds fall outside 2024 "
            f"(Chapter 4).",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic", color="#555555")
    fig.savefig(f"{OUT}/fig_metric_timeline.png")
    plt.close(fig)
    print("fig_metric_timeline.png")
    for scen in ("C0", "S1"):
        d = df[df.scen == scen]
        if d.empty:
            continue
        print(f"    {scen}: {len(d):,} windows, "
              f"D mean {d.metric_value.mean():.4f}, max {d.metric_value.max():.4f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_window_pilot()
    fig_recall()
    fig_magnitude_ratio()
    fig_classification()
    fig_decision_tree()
    fig_psi_binning()
    fig_unseen_category()
    fig_metric_timeline()
    print(f"\nfigures written to {OUT}/")
