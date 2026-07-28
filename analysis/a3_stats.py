"""[A3-IMPROVEMENT] Statistical analysis for the back-off trigger study.

Produces the numbers the report's results/ablation/trade-off sections cite:
  1. Paired McNemar exact tests between arms (same items -> paired test).
  2. Bootstrap 95% CIs on every arm's accuracy (the noise floor).
  3. Per-signal fire rate + precision/recall against the oracle correctness label.
  4. Cost/accuracy trade-off table.

Writes CSVs to results/summary/ for the LaTeX tables to read.
"""

import json
import os
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import binomtest

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "a3")
SUMMARY_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "summary")


def load_arm(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return pd.DataFrame(rows)


def mcnemar_exact(a: pd.Series, b: pd.Series) -> dict:
    """Exact McNemar test on paired binary outcomes.

    The arms are evaluated on identical items, so correctness is paired; a
    two-proportion z-test would ignore that pairing and overstate the
    variance. Only the discordant pairs carry information.
    """
    both = pd.concat([a, b], axis=1).dropna()
    x, y = both.iloc[:, 0].astype(int), both.iloc[:, 1].astype(int)
    n01 = int(((x == 0) & (y == 1)).sum())  # b fixed what a got wrong
    n10 = int(((x == 1) & (y == 0)).sum())  # b broke what a got right
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "delta_pp": 0.0}
    p = binomtest(n01, n_disc, 0.5).pvalue
    return {
        "n01": n01,
        "n10": n10,
        "p_value": round(float(p), 5),
        "delta_pp": round(100 * (y.mean() - x.mean()), 2),
    }


def bootstrap_ci(correct: pd.Series, n_boot: int = 10000, seed: int = 233) -> tuple:
    """Percentile bootstrap 95% CI on accuracy.

    Makes the small-n noise floor explicit -- the whole reason the n=10
    Assignment-2 trends were uninterpretable.
    """
    arr = correct.dropna().astype(int).to_numpy()
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return (round(100 * float(np.percentile(means, 2.5)), 2),
            round(100 * float(np.percentile(means, 97.5)), 2))


def signal_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Fire rate and precision/recall of each signal against the oracle
    'ReAct was actually wrong' label.

    This is the mechanistic evidence: a signal can raise accuracy simply by
    backing off more often, so fire rate alone proves nothing. Precision
    (fired AND was wrong / fired) shows the signal is selective.
    """
    if "signals" not in df.columns:
        return pd.DataFrame()
    sig = pd.json_normalize(df["signals"])
    wrong = (df["correct"] == 0).reset_index(drop=True)
    out = []
    for name in ["s1_exhausted", "s2_thin_evidence", "s3_unsupported"]:
        if name not in sig:
            continue
        fired = sig[name].astype(bool).reset_index(drop=True)
        tp = int((fired & wrong).sum())
        fp = int((fired & ~wrong).sum())
        out.append({
            "signal": name,
            "fire_rate": round(100 * fired.mean(), 1),
            "precision": round(100 * tp / (tp + fp), 1) if (tp + fp) else float("nan"),
            "recall": round(100 * tp / wrong.sum(), 1) if wrong.sum() else float("nan"),
            "n_fired": int(fired.sum()),
        })
    # The union (full CGA) as an extra row.
    cols = [c for c in ["s1_exhausted", "s2_thin_evidence", "s3_unsupported"] if c in sig]
    if cols:
        fired = sig[cols].astype(bool).any(axis=1).reset_index(drop=True)
        tp = int((fired & wrong).sum())
        fp = int((fired & ~wrong).sum())
        out.append({
            "signal": "cga_union",
            "fire_rate": round(100 * fired.mean(), 1),
            "precision": round(100 * tp / (tp + fp), 1) if (tp + fp) else float("nan"),
            "recall": round(100 * tp / wrong.sum(), 1) if wrong.sum() else float("nan"),
            "n_fired": int(fired.sum()),
        })
    return pd.DataFrame(out)


def build_summary() -> pd.DataFrame:
    rows = []
    for path in sorted(glob(os.path.join(RAW_DIR, "*.jsonl"))):
        df = load_arm(path)
        if df.empty:
            continue
        name = os.path.basename(path).replace(".jsonl", "")
        lo, hi = bootstrap_ci(df["correct"])
        rows.append({
            "file": name,
            "domain": df["domain"].iloc[0],
            "method": df["method"].iloc[0],
            "trigger": df["trigger"].iloc[0] if "trigger" in df else "paper",
            "n": len(df),
            "accuracy": round(100 * df["correct"].mean(), 2),
            "ci_lo": lo,
            "ci_hi": hi,
            "backoff_rate": round(100 * df["backoff_triggered"].mean(), 1)
                            if "backoff_triggered" in df else 0.0,
            "mean_calls": round(df["n_calls"].dropna().mean(), 2) if "n_calls" in df else float("nan"),
            "cost_usd": round(df["cost_usd"].sum(), 4) if "cost_usd" in df else float("nan"),
            "cost_per_q": round(df["cost_usd"].mean(), 5) if "cost_usd" in df else float("nan"),
            "errors": int(df["error"].notna().sum()) if "error" in df else 0,
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summary = build_summary()
    if summary.empty:
        print("No results in", RAW_DIR)
        return
    summary.to_csv(os.path.join(SUMMARY_DIR, "a3_summary.csv"), index=False)
    print(summary.to_string(index=False))

    # Signal diagnostics from the --diagnose ReAct runs.
    diag_rows = []
    for path in sorted(glob(os.path.join(RAW_DIR, "*react_n*.jsonl"))):
        df = load_arm(path)
        d = signal_diagnostics(df)
        if not d.empty:
            d.insert(0, "domain", df["domain"].iloc[0])
            diag_rows.append(d)
    if diag_rows:
        diag = pd.concat(diag_rows, ignore_index=True)
        diag.to_csv(os.path.join(SUMMARY_DIR, "a3_signal_diagnostics.csv"), index=False)
        print("\n=== Signal diagnostics ===")
        print(diag.to_string(index=False))

    # Paired McNemar: every trigger arm against the paper control, for both
    # hybrid directions.
    tests = []
    for method in ["react_cotsc", "cotsc_react"]:
        for domain in summary["domain"].unique():
            arms = summary[(summary["domain"] == domain) & (summary["method"] == method)]
            ctrl = arms[arms["trigger"] == "paper"]
            if ctrl.empty:
                continue
            base = load_arm(os.path.join(RAW_DIR, ctrl.iloc[0]["file"] + ".jsonl"))
            base = base.sort_values("index").set_index("index")["correct"]
            for _, arm in arms.iterrows():
                if arm["trigger"] == "paper":
                    continue
                other = load_arm(os.path.join(RAW_DIR, arm["file"] + ".jsonl"))
                other = other.sort_values("index").set_index("index")["correct"]
                res = mcnemar_exact(base, other)
                res.update({"domain": domain, "method": method, "arm": arm["trigger"], "vs": "paper"})
                tests.append(res)
    if tests:
        t = pd.DataFrame(tests)[["domain", "method", "arm", "vs", "delta_pp", "n01", "n10", "p_value"]]
        t.to_csv(os.path.join(SUMMARY_DIR, "a3_mcnemar.csv"), index=False)
        print("\n=== McNemar (paired, vs paper control) ===")
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
