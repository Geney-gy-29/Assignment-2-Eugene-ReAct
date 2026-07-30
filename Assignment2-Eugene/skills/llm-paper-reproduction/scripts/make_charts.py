"""Aggregate results/raw/*.jsonl and generate paper-comparison charts into analysis/figures/."""
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

METHOD_ORDER = ["standard", "cot", "act", "react", "cot_sc", "react_cotsc", "cotsc_react"]
METHOD_LABELS = {
    "standard": "Standard",
    "cot": "CoT",
    "act": "Act",
    "react": "ReAct",
    "cot_sc": "CoT-SC",
    "react_cotsc": "ReAct→CoT-SC",
    "cotsc_react": "CoT-SC→ReAct",
}

# Table 1, Yao et al. 2023 (PaLM-540B, arXiv:2210.03629) -- HotpotQA EM / Fever accuracy, %
PAPER = {
    "hotpotqa": {"standard": 28.7, "cot": 29.4, "act": 25.7, "react": 27.4,
                 "cot_sc": 33.4, "react_cotsc": 35.1, "cotsc_react": 34.2},
    "fever": {"standard": 57.1, "cot": 56.3, "act": 58.9, "react": 60.9,
              "cot_sc": 60.4, "react_cotsc": 62.0, "cotsc_react": 64.6},
}


def load_results():
    rows = []
    for path in glob.glob(os.path.join(RAW_DIR, "*.jsonl")):
        fname = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if not records:
            continue
        domain = records[0]["domain"]
        method = records[0]["method"]
        n = len(records)
        acc = sum(r["correct"] for r in records) / n * 100
        backoff_rate = sum(r.get("backoff_triggered", False) for r in records) / n * 100
        rows.append({"domain": domain, "method": method, "n": n, "accuracy": acc,
                      "backoff_rate": backoff_rate, "source": fname})
    return pd.DataFrame(rows)


def plot_domain_methods(df, domain):
    sub = df[df["domain"] == domain].set_index("method")
    methods = [m for m in METHOD_ORDER if m in sub.index]
    repro = [sub.loc[m, "accuracy"] for m in methods]
    paper = [PAPER[domain][m] for m in methods]
    labels = [METHOD_LABELS[m] for m in methods]

    x = range(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], paper, width, label="Paper (PaLM-540B)", color="#888888")
    ax.bar([i + width / 2 for i in x], repro, width, label=f"Reproduction (GLM-5.2, n={sub['n'].iloc[0]})", color="#4C72B0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    metric = "Exact Match" if domain == "hotpotqa" else "Label Accuracy"
    ax.set_title(f"{domain.upper()}: Paper vs. Reproduction ({metric})")
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{domain}_paper_vs_repro.png"), dpi=150)
    plt.close(fig)


def plot_backoff_rates(df):
    sub = df[df["method"].isin(["react_cotsc", "cotsc_react"])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, domain in enumerate(["hotpotqa", "fever"]):
        d = sub[sub["domain"] == domain].set_index("method")
        methods = [m for m in ["react_cotsc", "cotsc_react"] if m in d.index]
        vals = [d.loc[m, "backoff_rate"] for m in methods]
        x = [j + i * 0.35 for j in range(len(methods))]
        ax.bar(x, vals, width=0.35, label=domain.upper())
    ax.set_xticks([j + 0.175 for j in range(2)])
    ax.set_xticklabels(["ReAct→CoT-SC", "CoT-SC→ReAct"])
    ax.set_ylabel("Backoff trigger rate (%)")
    ax.set_title("Hybrid Strategy Backoff Rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "backoff_rates.png"), dpi=150)
    plt.close(fig)


def plot_domain_comparison(df):
    piv = df.pivot(index="method", columns="domain", values="accuracy")
    piv = piv.reindex([m for m in METHOD_ORDER if m in piv.index])
    fig, ax = plt.subplots(figsize=(9, 5))
    piv.rename(index=METHOD_LABELS).plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Reproduction: HotpotQA vs. FEVER Accuracy by Method")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "hotpotqa_vs_fever.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_results()
    if df.empty:
        raise SystemExit("No result files found in results/raw/")

    summary_dir = os.path.join(os.path.dirname(__file__), "..", "results", "summary")
    os.makedirs(summary_dir, exist_ok=True)
    df.sort_values(["domain", "method"]).to_csv(os.path.join(summary_dir, "accuracy_summary.csv"), index=False)

    for domain in ["hotpotqa", "fever"]:
        if domain in df["domain"].values:
            plot_domain_methods(df, domain)
    plot_backoff_rates(df)
    plot_domain_comparison(df)
    print(f"Wrote {len(df)} rows to results/summary/accuracy_summary.csv")
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
