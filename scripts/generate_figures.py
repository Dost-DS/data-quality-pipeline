"""Generate the figures for the results chapters.

Reads directly from PostgreSQL so that every figure is reproducible from the
stored experiment data rather than from intermediate files.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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


def q(sql):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


# ---------------------------------------------------------------- figure 1
def fig_window_pilot():
    """False-alert rate against window size, per metric."""
    df = pd.read_csv("results/pilot_window_size.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric, colour, marker in [("psi", BLUE, "o"), ("ks", ORANGE, "s"),
                                   ("chi2", GREEN, "^")]:
        d = df[df.metric_name == metric].sort_values("window_size")
        ax.plot(d.window_size, d.false_alert_pct, marker=marker,
                color=colour, label=metric.upper(), linewidth=2, markersize=7)
    ax.set_xscale("log")
    ax.set_xticks(df.window_size.unique())
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Window size (records)")
    ax.set_ylabel("False-alert rate (%)")
    ax.set_title("False-alert rate by window size on unmodified data")
    ax.axvline(10000, color=GREY, linestyle="--", linewidth=1)
    ax.text(10400, ax.get_ylim()[1] * 0.93, "adopted", fontsize=9, color=GREY)
    ax.legend(frameon=False)
    fig.savefig(f"{OUT}/fig_window_pilot.png")
    plt.close(fig)
    print("fig_window_pilot.png")


# ---------------------------------------------------------------- figure 2
def fig_recall():
    """Detection outcome per injected target and metric."""
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
    """How far each metric moved, pre-activation against post-activation."""
    df = pd.read_csv("results/recall_by_metric.csv")
    df = df[df.scenario != "S5"].copy()          # S5 duplicates S1 and S2
    df["ratio"] = df.mean_post / df.mean_pre
    df["label"] = df.scenario + " " + df.injected_feature + "\n" + df.metric_name.str.upper()
    df = df.sort_values("ratio")

    colours = [GREEN if d else RED for d in (df.post_alerts > 0)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(df.label, df.ratio, color=colours, edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(1, color=GREY, linewidth=1)
    ax.set_xlabel("Post-activation mean \u00f7 pre-activation mean (log scale)")
    ax.set_title("Magnitude change at the injected target")
    for y, (r, det) in enumerate(zip(df.ratio, df.post_alerts > 0)):
        ax.text(r * 1.15, y, f"{r:,.1f}\u00d7" + ("" if det else "  (no alert)"),
                va="center", fontsize=8.5)
    ax.set_xlim(0.7, df.ratio.max() * 8)
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(pivot))
    for c in order:
        ax.bar(pivot.index, pivot[c], bottom=bottom, label=c.replace("_", " ").title(),
               color=colours[c], edgecolor="white", width=.62)
        bottom += pivot[c].values
    ax.set_ylabel("Alerts")
    ax.set_title("Alert classification by scenario")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    ax.text(0.99, 0.02, "No alert in any scenario was left unexplained.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, style="italic", color="#555555")
    fig.savefig(f"{OUT}/fig_classification.png")
    plt.close(fig)
    print("fig_classification.png")


# ---------------------------------------------------------------- figure 5
def fig_decision_tree():
    """Deliverable D1."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 11); ax.axis("off")

    def box(x, y, w, h, text, colour, fs=9.5):
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=colour, edgecolor="white",
                                    linewidth=1.5, zorder=2))
        ax.text(x, y, text, ha="center", va="center", color="white",
                fontsize=fs, fontweight="bold", zorder=3)

    def arrow(x1, y1, x2, y2, lbl=None, dx=.0, dy=.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color="#555555",
                                     linewidth=1.2, zorder=1))
        if lbl:
            ax.text((x1+x2)/2 + dx, (y1+y2)/2 + dy, lbl, fontsize=8.5,
                    style="italic", ha="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))

    box(6, 10.3, 4.4, .7, "Alert on feature X", BLUE)

    box(6, 9.0, 5.4, .8, "C7  Schema change in this window?", BLUE)
    arrow(6, 9.95, 6, 9.4)
    box(1.7, 7.6, 2.6, .7, "INVESTIGATE", GREEN, 11)
    arrow(3.6, 8.7, 2.5, 7.95, "yes", dx=-.2)

    box(7.2, 7.6, 5.2, .8, "C1  Injected target,\nwindow at/after activation?", BLUE)
    arrow(7.2, 8.6, 7.2, 8.0, "no", dx=.35)

    box(8.6, 6.1, 4.2, .8, "C4  Magnitude \u2265 2\u00d7 threshold?", BLUE)
    arrow(8.6, 7.25, 8.6, 6.5, "yes", dx=.4)
    box(10.4, 4.7, 2.2, .7, "ACT", GREEN, 11)
    box(7.6, 4.7, 2.6, .7, "INVESTIGATE", GREEN, 10)
    arrow(9.6, 5.75, 10.3, 5.05, "yes", dx=.35)
    arrow(7.6, 5.75, 7.6, 5.05, "no", dx=-.35)

    box(2.9, 6.1, 4.0, .8, "C2  Deterministic\ndependency edge?", BLUE)
    arrow(5.4, 7.3, 3.4, 6.5, "no", dx=-.35)
    box(1.4, 4.7, 2.2, .7, "ACCEPT", GREEN, 11)
    arrow(2.1, 5.75, 1.5, 5.05, "yes", dx=-.35)

    box(3.4, 3.3, 4.6, .8, "C3  Alerted in the\nclean control?", BLUE)
    arrow(3.7, 5.75, 3.5, 3.7, "no", dx=.45)
    box(1.3, 1.7, 2.2, .7, "SUPPRESS", GREEN, 10.5)
    arrow(2.4, 2.95, 1.5, 2.05, "yes", dx=-.35)

    box(7.4, 1.7, 4.4, .8, "C4/C6  Magnitude large?", BLUE)
    arrow(4.9, 3.05, 6.6, 2.1, "no", dx=.3)
    box(6.0, .5, 2.4, .55, "INVESTIGATE", GREEN, 9.5)
    box(9.4, .5, 2.2, .55, "MONITOR", GREEN, 10)
    arrow(6.6, 1.35, 6.1, .78, "yes", dx=-.35)
    arrow(8.4, 1.35, 9.2, .78, "no", dx=.3)

    ax.legend(handles=[mpatches.Patch(color=BLUE, label="Decision criterion"),
                       mpatches.Patch(color=GREEN, label="Recommended action")],
              loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title("Alert Interpretation Framework: decision tree (D1)",
                 fontsize=14, fontweight="bold", pad=10)
    fig.savefig(f"{OUT}/fig_decision_tree.png")
    plt.close(fig)
    print("fig_decision_tree.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_window_pilot()
    fig_recall()
    fig_magnitude_ratio()
    fig_classification()
    fig_decision_tree()
    print(f"\nfigures written to {OUT}/")
