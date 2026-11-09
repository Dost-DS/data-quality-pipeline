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

  All ten figures  Redrawn for print (October 2026): no title or note
                     inside the image, since the thesis caption carries both;
                     sized for a 16 cm text width; colours unchanged. The
                     decision tree now shows that an unexplained chi-square
                     alert skips C4; the PSI figure states that its values
                     come from the pilot slice; the magnitude ratios use the
                     window_end boundary, so they match the Chapter 5 tables;
                     fig_replay_pipeline draws the runs at their recorded
                     start times instead of a hand-written sequence.

  fig_replay_pipeline  Added when S7 was run. Previously hand-drawn and kept
                     outside this script, so the S7 row and the run/alert
                     counts had to be edited by hand and were not; this
                     version queries the eight runs and their alert counts
                     from the database instead.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.ticker
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
    if df.empty:
        print("  ! no PILOT rows in drift_scores; skipping fig_window_pilot")
        return
    df["pct"] = 100.0 * df.alerts / df.scores

    # KS is computed on numerical features only and chi-square on categorical
    # only, so neither needs splitting. Only PSI is applied to both.
    series = [("psi",  "numerical",   "PSI, numerical",   BLUE,   "o", "--", "white"),
              ("psi",  "categorical", "PSI, categorical", BLUE,   "o", "-",  BLUE),
              ("ks",   "numerical",   "KS, numerical",    ORANGE, "s", "-",  ORANGE),
              ("chi2", "categorical", "Chi-square, categorical", GREEN, "^", "-", GREEN)]

    fig, ax = plt.subplots(figsize=(TEXT_W, 3.0))
    for metric, ftype, label, colour, marker, style, mfc in series:
        d = df[(df.metric_name == metric) & (df.feature_type == ftype)].sort_values("window_size")
        if d.empty:
            continue
        ax.plot(d.window_size, d.pct, marker=marker, color=colour, label=label,
                linewidth=1.3, markersize=4.5, linestyle=style,
                markerfacecolor=mfc, markeredgewidth=1.0, zorder=3)
        print(f"    {label:<26}" + "  ".join(f"{x:6.2f}" for x in d.pct)
              + "   (alerts/scores: " + ", ".join(f"{a}/{s}" for a, s in zip(d.alerts, d.scores)) + ")")

    ax.set_xscale("log")
    ws = sorted(df.window_size.unique())
    ax.set_xticks(ws)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(ws[0] / 1.15, ws[-1] * 1.15)
    ax.set_ylim(-1.2, 36)
    ax.axvline(10000, color=GREY, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(10000 * 1.04, 35, "adopted", fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("Window size (records)")
    ax.set_ylabel("Baseline alert rate (%)")
    _print_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.86),
              handlelength=2.2)
    _save_print(fig, "fig_window_pilot")


# ---------------------------------------------------------------- figure 2
def fig_recall():
    """Detection outcome per injected target and metric."""
    df = pd.read_csv("results/recall_by_metric.csv")
    df["detected"] = df.post_alerts > 0
    df["row"] = df.scenario + "  " + df.injected_feature
    pivot = df.pivot_table(index="row", columns="metric_name",
                           values="detected", aggfunc="first")
    pivot = pivot[[c for c in ["psi", "ks", "chi2"] if c in pivot.columns]]
    names = {"psi": "PSI", "ks": "KS", "chi2": "Chi-square"}

    n_r, n_c = pivot.shape
    fig, ax = plt.subplots(figsize=(TEXT_W * 0.72, 0.3 * n_r + 0.5))
    for i in range(n_r):
        for j in range(n_c):
            v = pivot.iloc[i, j]
            if pd.isna(v):
                face, txt, tc = "#ECECEC", "not applied", "#8A8A8A"
            elif v:
                face, txt, tc = GREEN, "detected", "white"
            else:
                face, txt, tc = RED, "missed", "white"
            ax.add_patch(plt.Rectangle((j, n_r - i - 1), 1, 1,
                                       facecolor=face, edgecolor="white", linewidth=2))
            ax.text(j + .5, n_r - i - .5, txt, ha="center", va="center",
                    fontsize=7, color=tc, fontweight="bold")
    ax.set_xlim(0, n_c); ax.set_ylim(0, n_r)
    ax.set_xticks([j + .5 for j in range(n_c)])
    ax.set_xticklabels([names[c] for c in pivot.columns], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks([n_r - i - .5 for i in range(n_r)])
    ax.set_yticklabels(pivot.index, fontsize=7.5)
    ax.grid(False); ax.set_frame_on(False); ax.tick_params(length=0)
    _save_print(fig, "fig_recall")


# ---------------------------------------------------------------- figure 3
def fig_magnitude_ratio():
    """Post-activation magnitude against the control's pre-activation mean.

    A window counts as pre-activation when it ends before the activation
    point, the convention used throughout the thesis, so the ratios agree with
    the tables in Chapter 5. S5 repeats the S1 and S2 changes and S6 has its
    own figure, because its chi-square value follows from the floor on a zero
    expected count rather than from the data.
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
          AND d.window_end < TIMESTAMPTZ '{act}'
          AND d.metric_value IS NOT NULL
        GROUP BY 1, 2""")

    df = pd.read_csv("results/recall_by_metric.csv")
    df = df[~df.scenario.isin(["S5", "S6"])].copy()
    df = df.merge(ctrl, left_on=["injected_feature", "metric_name"],
                  right_on=["feature_name", "metric_name"], how="left")
    disagree = (df.control_pre_mean - df.mean_pre).abs() / df.mean_pre.abs()
    if (disagree > 0.001).any():
        print("    note: stored mean_pre differs from the control pre-activation mean")
        for _, r in df[disagree > 0.001].iterrows():
            print(f"      {r.scenario} {r.injected_feature}/{r.metric_name}: "
                  f"stored {r.mean_pre:,.4f}  control {r.control_pre_mean:,.4f}")

    names = {"psi": "PSI", "ks": "KS", "chi2": "chi-square"}
    df["ratio"] = df.mean_post / df.control_pre_mean
    df["label"] = df.scenario + " " + df.injected_feature + ", " + df.metric_name.map(names)
    df = df.sort_values("ratio")

    colours = [GREEN if d else RED for d in (df.post_alerts > 0)]
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.9))
    ax.barh(df.label, df.ratio, color=colours, edgecolor="white", height=0.7)
    ax.set_xscale("log")
    ax.axvline(1, color=GREY, linewidth=0.8)
    ax.set_xlabel("Post-activation mean divided by the control's pre-activation mean (log scale)")
    for y, (r, det) in enumerate(zip(df.ratio, df.post_alerts > 0)):
        ax.text(r * 1.15, y, f"{r:,.1f}×" + ("" if det else "  no alert"),
                va="center", fontsize=7, color=INK)
    ax.set_xlim(0.7, df.ratio.max() * 12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _print_axes(ax)
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    _save_print(fig, "fig_magnitude_ratio")
    for _, r in df.iterrows():
        print(f"    {r.label:<34} {r.ratio:8.1f}x")


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

    fig, ax = plt.subplots(figsize=(TEXT_W, 2.9))
    bottom = np.zeros(len(pivot))
    for c in order:
        ax.bar(pivot.index, pivot[c], bottom=bottom,
               label=c.replace("_", " ").capitalize(),
               color=colours[c], edgecolor="white", linewidth=0.6, width=.62)
        bottom += pivot[c].values
    ax.set_ylabel("Alerts")
    ax.set_ylim(0, bottom.max() * 1.22)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _print_axes(ax)
    ax.tick_params(axis="x", length=0, labelsize=8, colors=INK)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper left", handlelength=1.2)
    _save_print(fig, "fig_classification")
    print(pivot.astype(int).to_string())


# ===================================================================== PRINT
# The four functions below draw the figures that appear in the thesis body.
# They follow one set of rules so that the figures read as a set:
#   * no title and no explanatory note inside the image; the thesis caption
#     carries both
#   * sized for a 16 cm text width, 7 to 8 pt type, thin lines
#   * the colours of the earlier versions, so the figures stay recognisable
TEXT_W = 6.3          # inches, 16 cm
INK, MUTED = "#1A1A1A", "#555555"
PURPLE = "#7B5EA7"


def _print_axes(ax):
    """Recessive axes for a printed chart."""
    ax.grid(False)
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
        ax.spines[s].set_color(MUTED)
    ax.tick_params(labelsize=7.5, colors=MUTED, width=0.6)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)


def _save_print(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(name + ".png")


class _Diagram:
    """A drawing surface measured in centimetres, so that boxes and type keep
    their size when the figure is placed at the full text width."""

    def __init__(self, w_cm, h_cm, x0=0.0, y0=0.0):
        self.fig = plt.figure(figsize=(w_cm / 2.54, h_cm / 2.54))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(x0, x0 + w_cm)
        self.ax.set_ylim(y0, y0 + h_cm)
        self.ax.axis("off")

    def box(self, x, y, w, h, text, face, edge, tc=INK, live=True, bold=False,
            fs=7.5, square=False):
        self.ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="square,pad=0" if square else "round,pad=0,rounding_size=0.12",
            facecolor=face if live else PALE, edgecolor=edge if live else "#5A5A5A",
            linewidth=0.9, linestyle="-" if live else (0, (3, 2)), zorder=2))
        self.ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                     color=tc if live else "#242424",
                     fontweight="bold" if bold else "normal",
                     linespacing=1.25, zorder=3)
        return (x, y, w, h)

    def arrow(self, p1, p2, live=True, rad=0.0, both=False):
        self.ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle=("<|-|>" if both else "-|>") + ",head_length=3.2,head_width=1.8",
            color=INK if live else "#9A9A9A", linewidth=0.8,
            linestyle="-" if live else (0, (3, 2)),
            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=1))

    def line(self, pts, live=True):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=INK if live else "#9A9A9A", linewidth=0.8,
                     linestyle="-" if live else (0, (3, 2)), zorder=1)

    def label(self, x, y, text, live=True, ha="left"):
        self.ax.text(x, y, text, fontsize=7, ha=ha, va="center",
                     color=MUTED if live else "#9A9A9A", zorder=4)


def _E(b): return (b[0] + b[2] / 2, b[1])
def _W(b): return (b[0] - b[2] / 2, b[1])
def _N(b): return (b[0], b[1] + b[3] / 2)
def _S(b): return (b[0], b[1] - b[3] / 2)


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
          chi-square                       -> MONITOR
          magnitude >= 2x threshold        -> INVESTIGATE
          otherwise                        -> MONITOR

    Which paths were exercised is read from the stored evidence, so a path
    no alert took is drawn dashed rather than assumed.
    """
    try:
        ex = q("""SELECT
              (SELECT COUNT(*) FROM dq_monitoring.schema_snapshots
               WHERE has_schema_change)     AS schema_changes,
              (SELECT COUNT(*) FROM dq_monitoring.alert_classifications
               WHERE label = 'UNEXPLAINED') AS unexplained,
              (SELECT COUNT(*) FROM dq_monitoring.alert_classifications
               WHERE evidence->>'rule' = 'empirical dependency edge, absent from control')
                                            AS empirical_accept,
              (SELECT COUNT(*) FROM dq_monitoring.alert_classifications
               WHERE evidence->>'rule' = 'empirical edge, but the control also alerted here')
                                            AS empirical_suppress""")
        saw_schema = int(ex.schema_changes.iloc[0]) > 0
        saw_unexpl = int(ex.unexplained.iloc[0]) > 0
        saw_emp_accept = int(ex.empirical_accept.iloc[0]) > 0
        saw_emp_suppress = int(ex.empirical_suppress.iloc[0]) > 0
    except Exception as exc:
        # Drawing every path as exercised would overstate coverage, so skip.
        print(f"  ! could not query exercised branches ({type(exc).__name__}); "
              "skipping fig_decision_tree")
        return
    saw_empirical = saw_emp_accept or saw_emp_suppress

    d = _Diagram(16, 13.0)
    W_, H_, AW, AH = 4.8, 0.95, 2.6, 0.62
    X0, X1, X2, X3 = 2.7, 7.45, 10.75, 14.25

    def crit(x, y, w, h, text, live=True, fs=7.5):
        return d.box(x, y, w, h, text, BLUE, BLUE, tc="white", live=live, fs=fs)

    def action(x, y, text, live=True, w=AW):
        return d.box(x, y, w, AH, text, GREEN, GREEN, tc="white", live=live, bold=True)

    start = crit(X0, 12.4, W_, 0.55, "Alert on one feature and metric")
    c7 = crit(X0, 11.1, W_, H_, "C7  Schema change recorded\nfor this window?")
    inv7 = action(X1, 11.1, "INVESTIGATE", live=saw_schema)
    d.arrow(_S(start), _N(c7))
    d.arrow(_E(c7), _W(inv7), live=saw_schema)
    d.label(5.4, 11.28, "yes", live=saw_schema)

    c1 = crit(X0, 9.3, W_, H_, "C1  Injected target, window\nat or after activation?")
    chi = crit(X1, 9.3, 3.0, H_, "Metric is\nchi-square?")
    act = action(X3, 9.3, "ACT", w=2.2)
    c4 = crit(X2, 8.0, 3.2, H_, "C4  Value at least twice\nthe threshold?")
    inv4 = action(X3, 6.95, "INVESTIGATE")
    d.arrow(_S(c7), _N(c1)); d.label(X0 + 0.15, 10.2, "no")
    d.arrow(_E(c1), _W(chi)); d.label(5.4, 9.48, "yes")
    d.arrow(_E(chi), _W(act)); d.label(10.3, 9.48, "yes")
    d.line([_S(chi), (X1, 8.0)]); d.arrow((X1, 8.0), _W(c4)); d.label(X1 + 0.15, 8.65, "no")
    d.line([_E(c4), (X3, 8.0)]); d.arrow((X3, 8.0), _S(act)); d.label(X3 + 0.15, 8.65, "yes")
    d.line([_S(c4), (X2, 6.95)]); d.arrow((X2, 6.95), _W(inv4)); d.label(X2 + 0.15, 7.2, "no")

    c2 = crit(X0, 5.75, W_, H_, "C2  Declared edge from an injected\nfeature, at or after activation?", fs=7.2)
    edge = crit(X1, 5.75, 3.0, AH, "Edge type?")
    acc = action(X3, 5.75, "ACCEPT")
    c3e = crit(X2, 4.45, 3.2, H_, "C3  Same pair alerted\nin the control?", live=saw_empirical)
    sup2 = action(X3, 3.35, "SUPPRESS", live=saw_emp_suppress)
    d.arrow(_S(c1), _N(c2)); d.label(X0 + 0.15, 7.5, "no")
    d.arrow(_E(c2), _W(edge)); d.label(5.4, 5.93, "yes")
    d.arrow(_E(edge), _W(acc)); d.label(9.4, 5.93, "deterministic")
    d.line([_S(edge), (X1, 4.45)], live=saw_empirical)
    d.arrow((X1, 4.45), _W(c3e), live=saw_empirical)
    d.label(X1 + 0.15, 5.0, "empirical", live=saw_empirical)
    d.line([_E(c3e), (X3, 4.45)], live=saw_emp_accept)
    d.arrow((X3, 4.45), _S(acc), live=saw_emp_accept)
    d.label(X3 + 0.15, 4.85, "no", live=saw_emp_accept)
    d.line([_S(c3e), (X2, 3.35)], live=saw_emp_suppress)
    d.arrow((X2, 3.35), _W(sup2), live=saw_emp_suppress)
    d.label(X2 + 0.15, 3.7, "yes", live=saw_emp_suppress)

    c3 = crit(X0, 2.75, W_, H_, "C3  Same feature and metric\nalerted in the control?")
    sup = action(X1, 2.75, "SUPPRESS")
    d.arrow(_S(c2), _N(c3)); d.label(X0 + 0.15, 4.25, "no")
    d.arrow(_E(c3), _W(sup)); d.label(5.4, 2.93, "yes")

    c4u = crit(X0, 1.0, W_, H_, "C4  Value at least twice the\nthreshold? (chi-square: no)",
               live=saw_unexpl)
    invu = action(X1, 1.45, "INVESTIGATE", live=saw_unexpl)
    mon = action(X1, 0.45, "MONITOR", live=saw_unexpl)
    d.arrow(_S(c3), _N(c4u), live=saw_unexpl)
    d.label(X0 + 0.15, 1.88, "no (unexplained)", live=saw_unexpl)
    d.arrow((X0 + W_ / 2, 1.2), _W(invu), live=saw_unexpl)
    d.label(5.55, 1.5, "yes", live=saw_unexpl)
    d.arrow((X0 + W_ / 2, 0.8), _W(mon), live=saw_unexpl)
    d.label(5.55, 0.45, "no", live=saw_unexpl)

    handles = [mpatches.Patch(color=BLUE, label="Decision criterion"),
               mpatches.Patch(color=GREEN, label="Recommended action")]
    if not (saw_schema and saw_unexpl and saw_emp_accept and saw_emp_suppress):
        handles.append(Line2D([0], [0], color="#5A5A5A", linestyle=(0, (3, 2)),
                              linewidth=0.9, label="Path taken by no alert"))
    d.ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5,
                bbox_to_anchor=(0.99, 0.0))
    _save_print(d.fig, "fig_decision_tree")
    print(f"    schema branch exercised: {saw_schema}; "
          f"unexplained branch exercised: {saw_unexpl}; "
          f"empirical edge -> ACCEPT: {saw_emp_accept}; "
          f"empirical edge -> SUPPRESS: {saw_emp_suppress}")



# ---------------------------------------------------------------- figure 5b
def fig_replay_pipeline():
    """Order of execution of the registered runs (Appendix A, Figure A.1).

    One row per run in the order the runs were started, with the start date
    read from dq_monitoring.experiments. The five pilot replays are drawn as
    one row. The last row marks the classification and the framework, which
    were applied to all eight runs together after S7, so it carries no date.
    """
    df = q("""SELECT scenario, started_at
              FROM dq_monitoring.experiments
              ORDER BY started_at""")
    if df.empty:
        print("  ! no runs found; skipping fig_replay_pipeline")
        return
    df["day"] = pd.to_datetime(df.started_at, utc=True).dt.date

    main = df[df.scenario != "PILOT"]
    dups = main[main.duplicated("scenario", keep=False)]
    if not dups.empty:
        print("  ! more than one row per scenario; the latest is kept:",
              sorted(dups.scenario.unique()))
    main = main.drop_duplicates("scenario", keep="last")
    pilot = df[df.scenario == "PILOT"]

    what = {"C0": "clean control, no injection",
            "S1": "mean shift on fare_amount",
            "S2": "value substitution on payment_type",
            "S3": "rare-category shift on RatecodeID",
            "S4": "variance shift on trip_distance",
            "S5": "S1 and S2 combined",
            "S6": "unseen category on RatecodeID",
            "S7": "schema change, no distributional change"}
    PURPLE, CRIMSON, AMBER = "#7B5EA7", "#C0435A", "#D98A3D"
    no_injection = {"C0", "S7"}

    rows = [(r.started_at, r.day, r.scenario, what[r.scenario],
             CRIMSON if r.scenario in no_injection else GREEN)
            for r in main.itertuples()]
    if not pilot.empty:
        p = pilot.iloc[0]
        rows.append((p.started_at, p.day, "Pilot",
                     f"window-size pilot, {len(pilot)} window sizes on a "
                     "200,000-record slice", PURPLE))
    rows.sort(key=lambda r: pd.Timestamp(r[0]))

    n = len(rows) + 1
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * n + 0.9))
    ax.axis("off")
    ax.grid(False)
    y = [n - 1 - i for i in range(n)]
    x_date, x_line, x_name, x_what = 0.0, 1.05, 1.35, 2.05

    ax.plot([x_line, x_line], [y[-1], y[0]], color="#9a9a9a", lw=1.4, zorder=1)
    for i, (_, day, name, desc, colour) in enumerate(rows):
        ax.text(x_date + 0.85, y[i], f"{day.day} Sep", ha="right", va="center",
                fontsize=9.5, color="#333333")
        ax.scatter([x_line], [y[i]], s=95, color=colour, edgecolor="white",
                   linewidth=1.2, zorder=3)
        ax.text(x_name, y[i], name, ha="left", va="center", fontsize=10,
                fontweight="bold", color="#1a1a1a")
        ax.text(x_what, y[i], desc, ha="left", va="center", fontsize=9.5,
                color="#333333")
        if i > 0:
            gap = (day - rows[i - 1][1]).days
            if gap >= 5:
                ax.text(x_line + 0.14, (y[i] + y[i - 1]) / 2, f"{gap} days later",
                        ha="left", va="center", fontsize=8.5, style="italic",
                        color="#666666")

    ax.scatter([x_line], [y[-1]], s=95, marker="s", color=AMBER,
               edgecolor="white", linewidth=1.2, zorder=3)
    ax.text(x_date + 0.85, y[-1], "after S7", ha="right", va="center",
            fontsize=9.5, color="#333333")
    ax.text(x_name, y[-1], "Alert classification and framework, applied to "
            "all eight runs together", ha="left", va="center", fontsize=10,
            fontweight="bold", color="#1a1a1a")

    handles = [Line2D([], [], marker="o", ls="", ms=8, color=c, label=l)
               for c, l in [(CRIMSON, "No injection (C0, S7)"),
                            (GREEN, "Injection (S1 to S6)"),
                            (PURPLE, "Window-size pilot")]]
    handles.append(Line2D([], [], marker="s", ls="", ms=8, color=AMBER,
                          label="Analysis of all runs"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=4, frameon=False, fontsize=9, handletextpad=0.3,
              columnspacing=1.4)
    ax.set_xlim(-0.1, 7.6)
    ax.set_ylim(-0.6, n - 0.4)
    fig.savefig(f"{OUT}/fig_replay_pipeline.png")
    plt.close(fig)
    print("fig_replay_pipeline.png")
    for r in rows:
        print(f"    {r[1]}  {r[2]}")


# ---------------------------------------------------------------- figure 5c
def fig_system_architecture():
    """The artifact as designed: ingest, monitor each window, store and track,
    and interpret after all runs.

    Static: it draws the design, not experiment results, so it takes no
    database query. Italic names are the adapter interfaces of Table 4.2.
    """
    # fill, outline and text colour per kind of component
    KIND = {"data":   ("#F2F0EB", "#5F5E5A", "#2C2C2A"),   # data and artifacts
            "run":    ("#E1F5EE", "#0F6E56", "#085041"),   # host process, during a run
            "docker": ("#E6F1FB", "#185FA5", "#0C447C"),   # Docker service
            "after":  ("#EEEDFE", "#534AB7", "#3C3489"),   # host process, after the runs
            "output": ("#FAEEDA", "#854F0B", "#633806")}   # output
    LINE, NOTE, RULE = "#444441", "#5F5E5A", "#B4B2A9"

    d = _Diagram(16.2, 8.7, x0=0.0, y0=1.75)
    ax = d.ax

    def comp(cx, cy, w, h, title, body=(), iface=None, kind="data"):
        face, edge, tc = KIND[kind]
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, linewidth=0.9, zorder=2))
        lines = [(title, "bold", "normal")] + [(b, "normal", "normal") for b in body]
        if iface:
            lines.append((iface, "normal", "italic"))
        step = 0.34
        y = cy + step * (len(lines) - 1) / 2
        for k, (t, weight, style) in enumerate(lines):
            ax.text(cx, y - k * step, t, ha="center", va="center", color=tc,
                    fontsize=7.4 if k == 0 else 6.8, fontweight=weight, style=style, zorder=3)
        return (cx, cy, w, h)

    def arrow(p1, p2, both=False):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=("<|-|>" if both else "-|>")
                                     + ",head_length=3.2,head_width=1.8", color=LINE,
                                     linewidth=0.8, shrinkA=0, shrinkB=0, zorder=1))

    def line(pts):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=LINE, linewidth=0.8, zorder=1)

    def group(x0, x1, y0, y1):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=0.2",
                                    facecolor="none", edgecolor="#888780", linewidth=0.7,
                                    linestyle=(0, (3, 2)), zorder=0))

    # column headers
    for x0, x1, t in [(0.1, 6.0, "Ingest"), (6.25, 9.65, "Monitor each window"),
                      (9.95, 12.75, "Store and track"), (13.1, 16.1, "Interpret, after all runs")]:
        ax.text((x0 + x1) / 2, 10.15, t, ha="center", va="center", fontsize=7.6,
                fontweight="bold", color=NOTE)
        ax.plot([x0, x1], [9.88, 9.88], color=RULE, linewidth=0.8)

    RA, RB = 8.85, 6.3
    # ingest
    src = comp(1.5, RA, 2.9, 1.45, "NYC TLC taxi data",
               ["2024, 12 monthly files", "41,169,720 records"], kind="data")
    prod = comp(1.5, RB, 2.9, 1.5, "Producer", ["per monthly file:", "sort, inject, publish"],
                kind="run")
    kafka = comp(4.6, RB, 2.7, 1.5, "Apache Kafka", ["single partition", "with ZooKeeper"],
                 "EventSource", kind="docker")
    scen = comp(3.0, 3.5, 5.9, 1.35, "Scenarios C0 to S7 (Table 3.1)",
                ["one per replay, change applied from", "row 20,584,860 (3 July 2024)"], kind="data")
    arrow(_S(src), _N(prod))
    arrow((1.5, 3.5 + 0.675), _S(prod))
    arrow(_E(prod), _W(kafka))

    # monitor each window
    cx = 7.95
    ref = comp(cx, RA, 3.1, 1.85, "Reference artifacts",
               ["1 to 7 January 2024", "bins and frequencies,", "KS sample, schema,",
                "expectation suite"], kind="data")
    group(6.25, 9.65, 3.6, 7.55)
    val = comp(cx, 6.85, 3.1, 1.05, "Validation", ["Great Expectations"], "Validator", kind="run")
    drf = comp(cx, 5.6, 3.1, 0.85, "Drift", ["PSI, KS, chi-square"], kind="run")
    sch = comp(cx, 4.35, 3.1, 1.05, "Schema", ["columns against", "the stored schema"], kind="run")
    ax.text(cx, 3.45, "three consumers, run in turn,\non windows of 10,000 records",
            ha="center", va="top", fontsize=6.6, style="italic", color=NOTE, linespacing=1.3)
    arrow(_S(ref), (cx, 7.55))
    arrow(_E(kafka), (6.25, RB))

    # store and track
    sx = 11.3
    air = comp(sx, RA, 2.6, 1.45, "Apache Airflow", ["hourly health check", "daily drift summary"],
               "Scheduler", kind="docker")
    pg = comp(sx, RB + 0.2, 2.6, 1.6, "PostgreSQL", ["dq_monitoring,", "seven tables"],
              "MetricStore", kind="docker")
    mlf = comp(sx, 3.5, 2.6, 1.25, "MLflow", ["per-window metrics"], "RunTracker", kind="docker")
    arrow((9.65, 6.85), (sx - 1.3, 6.85))
    arrow(_N(pg), _S(air))
    line([(cx + 1.55, 5.6), (sx, 5.6)])
    arrow((sx, 5.6), _N(mlf))

    # interpret, after all runs
    ix = 14.6
    group(13.1, 16.1, 5.4, 9.75)
    cls = comp(ix, RA, 2.75, 1.45, "Classification",
               ["four labels, from the", "control run and the", "dependency map"], kind="after")
    itp = comp(ix, RB, 2.75, 1.45, "Interpretation", ["framework,", "criteria C1 to C7"],
               kind="after")
    act = comp(ix, 3.5, 2.75, 1.45, "Action per alert",
               ["act, investigate,", "monitor, accept,", "suppress"], kind="output")
    arrow(_S(cls), _N(itp))
    arrow(_S(itp), _N(act))
    arrow((sx + 1.3, RB + 0.2), (13.1, RB + 0.2), both=True)

    # legend
    items = [("data", "Data and artifacts"), ("run", "Host process, during a run"),
             ("docker", "Docker service"), ("after", "Host process, after the runs"),
             ("output", "Output")]
    handles = [mpatches.Patch(facecolor=KIND[k][0], edgecolor=KIND[k][1], linewidth=0.9, label=l)
               for k, l in items]
    handles.append(Line2D([0], [0], linestyle="", label="$\\it{Italic}$: adapter interface"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6,
              frameon=False, fontsize=6.6, handlelength=1.4, columnspacing=1.0,
              handletextpad=0.4)
    _save_print(d.fig, "fig_system_architecture")


_NUM_WORDS = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
              10: "ten", 12: "twelve", 13: "thirteen"}


def _num_word(n):
    return _NUM_WORDS.get(n, str(n))


# ---------------------------------------------------------------- figure 6
def fig_psi_binning():
    """Maximum PSI per numerical feature against window size (pilot runs).

    The mechanism behind the numerical PSI result. Bin edges are drawn over
    the baseline's full observed range, so a feature whose range is set by an
    extreme outlier gets bins far wider than its own variability, and no
    plausible shift moves mass across a boundary. The values come from the
    pilot replays of the 200,000-record slice, not from the full control run.
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

    # Share of the reference falling in the single most populated bin. On the
    # pilot slice it orders the features exactly inversely to their maximum PSI.
    shares = {}
    try:
        import json
        b = json.load(open("great_expectations/reference_baseline.json"))
        for f, r in b["numerical"].items():
            c = r["hist_counts"]
            shares[f] = 100.0 * max(c) / sum(c)
    except Exception as exc:
        print(f"    (modal-bin shares unavailable: {type(exc).__name__})")

    order = sorted(df.feature_name.unique(),
                   key=lambda f: -df[df.feature_name == f].max_psi.max())
    palette = [BLUE, ORANGE, GREEN, RED, PURPLE]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(TEXT_W, 3.1))
    for i, feat in enumerate(order):
        d = df[df.feature_name == feat].sort_values("ws")
        lbl = feat
        if feat in shares:
            # four decimals for trip_distance: 99.9995% would round to 100.00%
            pct = f"{shares[feat]:.2f}" if shares[feat] < 99.99 else f"{shares[feat]:.4f}"
            lbl += f"  ({pct}%)"
        ax.plot(d.ws, d.max_psi.clip(lower=1e-5), marker=markers[i % 5],
                color=palette[i % 5], label=lbl, linewidth=1.3, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.5)

    ax.axhline(0.2, color=GREY, linestyle="--", linewidth=0.9)
    ax.text(df.ws.max(), 0.23, "alert threshold, PSI = 0.20", fontsize=7,
            color=MUTED, ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ws = sorted(df.ws.unique())
    ax.set_xticks(ws)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(ws[0] / 1.15, ws[-1] * 1.15)
    ax.set_ylim(1e-4, 0.6)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Window size (records)")
    ax.set_ylabel("Maximum PSI on the pilot slice")
    _print_axes(ax)
    ax.legend(title="Feature (share of reference in largest bin)", title_fontsize=7,
              fontsize=7, frameon=True, framealpha=1, edgecolor="white",
              loc="upper right", bbox_to_anchor=(1.0, 0.83), handlelength=2.0,
              borderpad=0.3, labelspacing=0.3)
    _save_print(fig, "fig_psi_binning")
    for feat in order:
        d = df[df.feature_name == feat].sort_values("ws")
        r = f"  modal share {shares[feat]:>9.4f}%" if feat in shares else ""
        print(f"    {feat:<18}" + "  ".join(f"{v:9.4f}" for v in d.max_psi) + r)



# ---------------------------------------------------------------- figure 7
def fig_unseen_category():
    """RatecodeID under three conditions: control, rare-category shift, unseen.

    Post-activation means use window_end, the same boundary as
    build_result_tables.py, so the values here agree with recall_by_metric.csv.
    """
    df = q("""
        SELECT LEFT(experiment_id, 2) AS scen, metric_name,
               AVG(metric_value) FILTER (
                   WHERE window_end >= (SELECT MIN(activation_timestamp)
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
              "S3": "S3\nrare category\n0.88% to 25%",
              "S6": "S6\nunseen category\n0% to 25%"}
    order = ["C0", "S3", "S6"]
    colours = [GREY, ORANGE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.9))
    for ax, metric, title, thr in [
            (axes[0], "chi2", "Chi-square statistic", None),
            (axes[1], "psi", "Population Stability Index", 0.2)]:
        vals = [float(df[(df.scen == s) & (df.metric_name == metric)].post.iloc[0])
                for s in order]
        ax.bar([labels[s] for s in order], vals, color=colours,
               edgecolor="white", width=.6)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8, fontweight="normal", color=INK)
        ax.set_ylabel("Post-activation mean (log scale)")
        ax.set_ylim(min(vals) / 6, max(vals) * (40 if metric == "chi2" else 6))
        if metric == "psi":
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for i, v in enumerate(vals):
            txt = f"{v:,.0f}" if v >= 100 else f"{v:.3f}"
            ax.text(i, v * 1.5, txt, ha="center", fontsize=7, color=INK)
        if thr is not None:
            ax.axhline(thr, color=GREY, linestyle="--", linewidth=0.8)
            ax.text(-0.42, thr * 1.25, "threshold", fontsize=7, color=MUTED, ha="left")
        _print_axes(ax)
        ax.tick_params(axis="x", labelsize=7, colors=INK, length=0)
    fig.tight_layout(w_pad=2.0)
    _save_print(fig, "fig_unseen_category")
    for _, r in df.sort_values(["metric_name", "scen"]).iterrows():
        print(f"    {r.scen}  {r.metric_name:<5}  post = {float(r.post):,.4f}")


# ---------------------------------------------------------------- figure 8
def fig_metric_timeline():
    """KS on fare_amount across the corpus: control against S1.

    The control is drawn for the whole year. S1 is drawn from the first window
    that ends at or after activation; before that its values are identical to
    the control's, because nothing differs until the activation row.
    Windows whose start falls outside 2024 (corrupted timestamps) are omitted.
    """
    act = pd.Timestamp(q("""SELECT MIN(activation_timestamp) AS ts
                            FROM dq_monitoring.experiments
                            WHERE activation_timestamp IS NOT NULL""").ts.iloc[0])

    df = q("""
        SELECT LEFT(experiment_id, 2) AS scen, window_start, window_end, metric_value
        FROM dq_monitoring.drift_scores
        WHERE feature_name = 'fare_amount' AND metric_name = 'ks'
          AND experiment_id LIKE ANY (ARRAY['C0%', 'S1%'])
          AND window_start >= TIMESTAMPTZ '2024-01-01'
          AND window_start <  TIMESTAMPTZ '2025-01-01'
        ORDER BY window_start""")
    if df.empty:
        print("  ! no fare_amount KS rows; skipping fig_metric_timeline")
        return

    c0 = df[df.scen == "C0"].sort_values("window_start")
    s1 = df[(df.scen == "S1") & (df.window_end >= act)].sort_values("window_start")

    fig, ax = plt.subplots(figsize=(TEXT_W, 2.9))
    for d, colour in ((c0, GREY), (s1, RED)):
        ax.plot(d.window_start, d.metric_value, color=colour, linewidth=0.35, alpha=0.4)
        ax.plot(d.window_start, d.metric_value.rolling(50, center=True, min_periods=1).mean(),
                color=colour, linewidth=1.4)

    ax.axhline(0.1, color=GREEN, linestyle=":", linewidth=1.0)
    ax.text(pd.Timestamp("2024-01-04"), 0.112, "effect-size gate, D = 0.1", fontsize=7,
            color=GREEN, va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
    ax.axvline(act, color=BLUE, linestyle="--", linewidth=0.9)
    ax.text(act + pd.Timedelta(days=3), 0.755, "activation, 3 July 2024", fontsize=7,
            color=BLUE, va="top")

    # direct labels in the right margin, level with each rolling mean
    end = pd.Timestamp("2025-01-05")
    ax.text(end, float(s1.metric_value.tail(50).mean()), "S1\nfare shift", fontsize=7.5,
            color=RED, va="center", ha="left", clip_on=False, linespacing=1.1)
    ax.text(end, float(c0.metric_value.tail(50).mean()), "C0\ncontrol", fontsize=7.5,
            color=MUTED, va="center", ha="left", clip_on=False, linespacing=1.1)

    ax.set_ylim(0, 0.77)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.set_xlabel("Window start, event time (2024)")
    ax.set_ylabel("Kolmogorov-Smirnov D")
    _print_axes(ax)
    _save_print(fig, "fig_metric_timeline")
    for name, d in (("C0", c0), ("S1 after activation", s1)):
        print(f"    {name}: {len(d):,} windows, D mean {d.metric_value.mean():.4f}, "
              f"max {d.metric_value.max():.4f}")


# ---------------------------------------------------------------- figure 2.1
def fig_conceptual_framework():
    """Conceptual framework of the study (Chapter 2, Figure 2.1).

    Three checks on the event-time stream. Validation failures are recorded
    per batch, distributional monitoring raises the alerts, and the schema
    change flag goes straight to the decision procedure (criterion C7). Each
    alert has one of four possible causes; baseline variation can only be
    established with a clean control replay. Static diagram, no database query.
    """
    PALE_B, PALE_G, PALE_O, PALE_GR, PALE_R = ("#DCE6F1", "#DFF0E3", "#FBE7D6",
                                              "#EDEDED", "#F6DADD")
    fig, ax = plt.subplots(figsize=(10.6, 3.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc, ec, fs=9, sfs=7.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2 + (0.1 if sub else 0), title, ha="center",
                va="center", fontsize=fs, fontweight="bold", color="#1a1a1a")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.13, sub, ha="center", va="center",
                    fontsize=sfs, color="#333333")

    def arrow(x1, y1, x2, y2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, color="#444444", lw=1.2,
                                     linestyle=ls))

    mid = 1.75
    box(0.05, mid - 0.35, 1.45, 0.7, "Stream", "ordered by event time",
        PALE_GR, GREY, sfs=7)

    val_y, dist_y, sch_y = 2.6, mid - 0.31, 0.3
    for title, sub, y in [
            ("Rule-based validation", "declared constraints per batch", val_y),
            ("Distributional monitoring",
             "PSI with KS or chi-square, effect-size gate", dist_y),
            ("Schema monitoring", "columns against a stored schema", sch_y)]:
        box(1.9, y, 3.0, 0.62, title, sub, PALE_B, BLUE, sfs=7)
        arrow(1.5, mid, 1.9, y + 0.31)

    box(5.3, val_y + 0.06, 1.35, 0.5, "Failures", "recorded per batch",
        PALE_GR, GREY, fs=8.4, sfs=6.8)
    arrow(4.9, val_y + 0.31, 5.3, val_y + 0.31)

    box(5.3, mid - 0.35, 1.35, 0.7, "Alert", "window differs", PALE_R, RED)
    arrow(4.9, mid, 5.3, mid)

    causes = [("Injected change", None, 2.62),
              ("Declared propagation", None, 2.0),
              ("Baseline variation", "requires a clean control replay", 1.38),
              ("Unexplained", None, 0.76)]
    ax.text(8.075, 3.32, "Four possible causes", ha="center", va="center",
            fontsize=8, style="italic", color="#555555")
    for title, sub, y in causes:
        box(7.05, y, 2.05, 0.5, title, sub, PALE_O, ORANGE, fs=8.2, sfs=6.4)
        arrow(6.65, mid, 7.05, y + 0.25)

    act_x, act_w = 9.5, 1.4
    box(act_x, mid - 0.35, act_w, 0.7, "Action", "decision procedure",
        PALE_G, GREEN, sfs=6.8)
    for _, _, y in causes:
        arrow(9.1, y + 0.25, act_x, mid)

    fx = act_x + act_w / 2
    ax.plot([4.9, fx], [sch_y + 0.31, sch_y + 0.31], color="#444444", lw=1.2,
            ls=(0, (4, 2)))
    arrow(fx, sch_y + 0.31, fx, mid - 0.35, ls=(0, (4, 2)))
    ax.text(6.0, sch_y + 0.22, "schema change flag", ha="center", va="top",
            fontsize=7.4, style="italic", color="#555555")

    fig.savefig(f"{OUT}/fig_conceptual_framework.png")
    plt.close(fig)
    print("fig_conceptual_framework.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_window_pilot()
    fig_recall()
    fig_magnitude_ratio()
    fig_classification()
    fig_decision_tree()
    fig_replay_pipeline()
    fig_system_architecture()
    fig_psi_binning()
    fig_unseen_category()
    fig_metric_timeline()
    fig_conceptual_framework()
    print(f"\nfigures written to {OUT}/")
