"""[A3-IMPROVEMENT] Figures for the ACM report.

Generates, into report/acm/figures/:
  fig_baseline.pdf     -- paper vs. A2(n=10) vs. A3(n=100) per domain
  fig_signals.pdf      -- per-signal fire rate / precision / recall
  fig_ablation.pdf     -- accuracy by trigger arm with bootstrap 95% CIs
  fig_tradeoff.pdf     -- accuracy vs. cost-per-question, Pareto frontier

PDF (vector) rather than PNG, since the report is LaTeX.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from a3_stats import RAW_DIR, SUMMARY_DIR, bootstrap_ci, load_arm  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "acm", "figures")

# Colour-blind-safe, prints legibly in greyscale.
C_PAPER = "#8c8c8c"
C_A2 = "#4C78A8"
C_A3 = "#F58518"
C_CGA = "#54A24B"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

# Reported in Yao et al. (2023), Tables 1 and 2 (PaLM-540B, PaLM-540B).
PAPER = {
    "hotpotqa": {"standard": 28.7, "cot": 29.4, "act": 25.7, "react": 27.4,
                 "cot_sc": 33.4, "react_cotsc": 35.1, "cotsc_react": 34.2},
    "fever": {"standard": 57.1, "cot": 56.3, "act": 58.9, "react": 60.9,
              "cot_sc": 60.4, "react_cotsc": 62.0, "cotsc_react": 64.6},
}
# Assignment-2 n=10 gate (results/summary/accuracy_summary.csv).
A2_N10 = {
    "hotpotqa": {"standard": 30.0, "cot": 60.0, "act": 70.0, "react": 60.0},
    "fever": {"standard": 20.0, "cot": 40.0, "act": 40.0, "react": 30.0},
}
LABELS = {"standard": "Standard", "cot": "CoT", "act": "Act", "react": "ReAct",
          "cot_sc": "CoT-SC", "react_cotsc": "ReAct->CoT-SC",
          "cotsc_react": "CoT-SC->ReAct"}
ARM_LABELS = {
    "paper": "S1 (paper)", "s1s2": "S1 or S2", "s1s3": "S1 or S3",
    "cga": "CGA tau=2", "cga_tau1": "CGA tau=1", "s3only": "S3 only",
}


def _summary() -> pd.DataFrame:
    return pd.read_csv(os.path.join(SUMMARY_DIR, "a3_summary.csv"))


def fig_baseline(summary: pd.DataFrame):
    """Paper vs. the n=10 gate vs. the repaired n=100 run.

    The n=10 -> n=100 movement is the point: it shows how much of the
    Assignment-2 'trend' was truncation bug + sampling noise.
    """
    base = summary[summary["method"].isin(A2_N10["fever"].keys())]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, domain in zip(axes, ["hotpotqa", "fever"]):
        d = base[base["domain"] == domain]
        d = d[d["trigger"] == "paper"] if "trigger" in d else d
        methods = [m for m in ["standard", "cot", "act", "react"]
                   if m in set(d["method"])]
        x = range(len(methods))
        w = 0.27
        paper = [PAPER[domain][m] for m in methods]
        a2 = [A2_N10[domain][m] for m in methods]
        rows = d.set_index("method")
        a3 = [rows.loc[m, "accuracy"] for m in methods]
        err = [[rows.loc[m, "accuracy"] - rows.loc[m, "ci_lo"] for m in methods],
               [rows.loc[m, "ci_hi"] - rows.loc[m, "accuracy"] for m in methods]]
        ax.bar([i - w for i in x], paper, w, label="Paper (PaLM-540B)", color=C_PAPER)
        ax.bar(list(x), a2, w, label="A2 repro ($n$=10)", color=C_A2)
        ax.bar([i + w for i in x], a3, w, yerr=err, capsize=2,
               label="A3 repaired ($n$=100)", color=C_A3,
               error_kw={"lw": 0.8, "ecolor": "#333"})
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABELS[m] for m in methods], rotation=15)
        ax.set_title("HotpotQA (EM)" if domain == "hotpotqa" else "FEVER (Acc)")
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_baseline.pdf"))
    plt.close(fig)


def fig_signals():
    """Fire rate vs. precision per signal.

    The load-bearing chart for the mechanism claim: a signal can lift accuracy
    just by firing constantly, so fire rate must be read against precision.
    """
    path = os.path.join(SUMMARY_DIR, "a3_signal_diagnostics.csv")
    if not os.path.exists(path):
        return
    diag = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, domain in zip(axes, ["hotpotqa", "fever"]):
        d = diag[diag["domain"] == domain]
        if d.empty:
            continue
        sigs = list(d["signal"])
        x = range(len(sigs))
        w = 0.38
        ax.bar([i - w / 2 for i in x], d["fire_rate"], w, label="Fire rate", color=C_A2)
        ax.bar([i + w / 2 for i in x], d["precision"], w, label="Precision", color=C_CGA)
        ax.axhline(50, ls=":", lw=0.8, color="#999")
        ax.set_xticks(list(x))
        ax.set_xticklabels([s.replace("_", "\n") for s in sigs], fontsize=6.5)
        ax.set_title("HotpotQA" if domain == "hotpotqa" else "FEVER")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("%")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_signals.pdf"))
    plt.close(fig)


def fig_ablation(summary: pd.DataFrame):
    """Accuracy per trigger arm with bootstrap 95% CIs."""
    d = summary[summary["method"] == "react_cotsc"]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, domain in zip(axes, ["hotpotqa", "fever"]):
        dd = d[d["domain"] == domain].sort_values("accuracy")
        if dd.empty:
            continue
        y = range(len(dd))
        colors = [C_PAPER if t == "paper" else (C_CGA if "cga" in str(t) else C_A3)
                  for t in dd["trigger"]]
        err = [dd["accuracy"] - dd["ci_lo"], dd["ci_hi"] - dd["accuracy"]]
        ax.barh(list(y), dd["accuracy"], xerr=err, color=colors, capsize=2,
                error_kw={"lw": 0.8, "ecolor": "#333"})
        ax.set_yticks(list(y))
        ax.set_yticklabels([ARM_LABELS.get(t, t) for t in dd["trigger"]], fontsize=7)
        ax.set_title("HotpotQA" if domain == "hotpotqa" else "FEVER")
        ax.set_xlabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_ablation.pdf"))
    plt.close(fig)


def fig_tradeoff(summary: pd.DataFrame):
    """Accuracy vs. cost per question, with the Pareto frontier.

    The required accuracy-vs-computational-cost analysis. A trigger that fires
    on everything collapses into plain CoT-SC: it would sit far right at
    similar accuracy, which this chart makes visible.
    """
    d = summary[summary["cost_per_q"].notna()]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, domain in zip(axes, ["hotpotqa", "fever"]):
        dd = d[d["domain"] == domain]
        if dd.empty:
            continue
        for _, r in dd.iterrows():
            is_arm = r["method"] == "react_cotsc"
            ax.scatter(r["cost_per_q"] * 100, r["accuracy"], s=34,
                       color=C_CGA if is_arm and "cga" in str(r["trigger"])
                       else (C_A3 if is_arm else C_A2),
                       marker="o" if is_arm else "^", zorder=3)
            lab = ARM_LABELS.get(r["trigger"], r["trigger"]) if is_arm else LABELS.get(r["method"], r["method"])
            ax.annotate(lab, (r["cost_per_q"] * 100, r["accuracy"]),
                        fontsize=5.5, xytext=(3, 3), textcoords="offset points")
        # Pareto frontier: cheapest point at or above each accuracy level.
        pts = sorted(zip(dd["cost_per_q"] * 100, dd["accuracy"]))
        front, best = [], -1
        for cx, cy in pts:
            if cy > best:
                front.append((cx, cy))
                best = cy
        if len(front) > 1:
            ax.plot([p[0] for p in front], [p[1] for p in front],
                    ls="--", lw=0.8, color="#999", zorder=1)
        ax.set_xlabel("Cost per question (US cents)")
        ax.set_title("HotpotQA" if domain == "hotpotqa" else "FEVER")
    axes[0].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_tradeoff.pdf"))
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    summary = _summary()
    fig_baseline(summary)
    fig_signals()
    fig_ablation(summary)
    fig_tradeoff(summary)
    print("Figures written to", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    main()
