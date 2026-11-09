"""Collect the fine-tuning runs into tables, and print the post's statistics.

    python -m scripts.results collect   # read every finished run under $NANOCHAT_BASE_DIR/lora_runs (and
                                        # lora_runs_ckpt) into results/runs.csv and results/curves.parquet, and
                                        # runs/pretrain.sh's logs into results/pretraining.csv
    python -m scripts.results report    # print the post's tables from results/runs.csv

results/ in the repository already holds these tables for the post's runs, so `report` and scripts/visualization/ work
without retraining; `collect` overwrites them with your own runs.

Metric. Every --eval-every steps, lora_sft scores the next slice of the validation set (--eval-batches micro-batches,
160 examples), cycling through it in a fixed order. PASS_WINDOW = 8 consecutive evals cover every validation set
once (1,224 / 1,319 / 1,000 examples), so a run's metric, `pass_floor`, is the lowest mean over 8 consecutive
evals: its best full-pass validation loss, as with early stopping.

Protocol. Each method takes its best grid value (drop rate, or weight decay) separately for every
(dataset, base, seed). Methods are then compared in pairs that share the dataset, base and seed, so the 15 pairs
per base (3 datasets x 5 seeds) share adapter initialization, data order and validation split. Tests are one-sided
Wilcoxon signed-rank tests in the direction "DLP is better"; confidence intervals are t-based.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from nanochat.common import get_base_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
RUNS_CSV = RESULTS / "runs.csv"
CURVES = RESULTS / "curves.parquet"  # keyed by (runs_dir, run): reruns in lora_runs_ckpt reuse run names

DATASETS = ["opencode", "gsm8k", "glaive"]
BASES = ["standard", "droppath"]
SEEDS = [0, 1, 2, 3, 4]
RATES = [0.05, 0.1, 0.2, 0.3]
WEIGHT_DECAYS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
PASS_WINDOW = 8
DLP = "L5"
METHODS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "WD"]
METHOD_NAMES = {
    "L0": "no regularization", "L1": "dropout(x), standard", "L2": "dropout(Ax)", "L3": "dropout(BAx)",
    "L4": "token DLP", "L5": "sequence DLP", "L6": "drop-path(Wx+BAx)", "WD": "weight decay",
}
MAIN = dict(lora_targets="c_q,c_k,c_v,c_proj,c_fc", rank=32, lr=3e-3, effective_batch_size=16, num_iterations=4500)


# ---------------------------------------------------------------------------
# Collection: one row per run, plus the validation and training curves

def run_row(d, runs_dir):
    cfg = json.loads((d / "summary.json").read_text())["config"]
    tl = pd.read_parquet(d / "train_log.parquet", columns=["kind", "step", "train_loss", "val_loss"])
    ev = tl[tl["kind"] == "eval"].dropna(subset=["val_loss"])
    tr = tl[tl["kind"] == "train"].dropna(subset=["train_loss"])
    vals = ev["val_loss"].to_numpy()
    i_min = int(np.argmin(vals))
    rolled = pd.Series(vals).rolling(PASS_WINDOW).mean().to_numpy()
    i_pass = int(np.nanargmin(rolled))
    tr_smooth = tr.set_index("step")["train_loss"].rolling(50, min_periods=1).mean()  # 50-step trailing mean
    step_min = int(ev["step"].iloc[i_min])
    row = dict(run=d.name, runs_dir=runs_dir, dataset=cfg["dataset"], base=cfg["base_model_tag"],
               rung=cfg["rung"], rate=cfg["rate"], weight_decay=cfg["weight_decay"], schedule=cfg["schedule"],
               noise=cfg["noise"], seed=cfg["seed"], lora_targets=cfg["lora_targets"], rank=cfg["rank"],
               alpha=cfg["alpha"], lr=cfg["lr"], effective_batch_size=cfg["effective_batch_size"],
               num_iterations=cfg["num_iterations"],
               pass_floor=float(rolled[i_pass]), step_pass=int(ev["step"].iloc[i_pass]),
               min_val=float(vals.min()), step_min=step_min, floor=float(np.sort(vals)[:5].mean()),
               final_val=float(vals[-1]),
               train_at_min=float(tr_smooth.asof(step_min)) if step_min > 0 else np.nan)
    curve = pd.DataFrame({"runs_dir": runs_dir, "run": d.name, "step": ev["step"].to_numpy(), "val_loss": vals})
    curve["train_loss"] = [float(tr_smooth.asof(s)) if s > 0 else np.nan for s in curve["step"]]
    return row, curve


def collect(runs_dirs=("lora_runs", "lora_runs_ckpt")):
    rows, curves = [], []
    for runs_dir in runs_dirs:
        root = Path(get_base_dir()) / runs_dir
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if (d / "summary.json").exists() and (d / "train_log.parquet").exists():
                row, curve = run_row(d, runs_dir)
                rows.append(row)
                curves.append(curve)
    RESULTS.mkdir(exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(RUNS_CSV, index=False)
    pd.concat(curves).to_parquet(CURVES, index=False)
    print(f"collected {len(runs)} runs -> {RUNS_CSV}, {CURVES}")
    collect_pretraining()
    return runs


def collect_pretraining():
    """results/pretraining.csv (Figure A1) from runs/pretrain.sh's logs, $NANOCHAT_BASE_DIR/pretrain_<base>.log."""
    logs = {base: Path(get_base_dir()) / f"pretrain_{base}.log" for base in BASES}
    if not all(log.exists() for log in logs.values()):
        print("pretraining logs not found; results/pretraining.csv unchanged")
        return
    rows = []
    for base, log in logs.items():
        text = log.read_text()
        for s, v in re.findall(r"^step (\d+)/\d+ .*?\| loss: ([\d.]+)", text, re.M):
            rows.append(dict(base=base, kind="train_loss", step=int(s), value=float(v)))
        for s, v in re.findall(r"^Step (\d+) \| Validation bpb: ([\d.]+)", text, re.M):
            rows.append(dict(base=base, kind="val_bpb", step=int(s), value=float(v)))
    pd.DataFrame(rows).to_csv(RESULTS / "pretraining.csv", index=False)
    print(f"collected pretraining curves -> {RESULTS / 'pretraining.csv'}")


def load():
    runs = pd.read_csv(RUNS_CSV)
    runs["method"] = np.where(runs["weight_decay"] > 0, "WD", runs["rung"])
    # the grid value a method is tuned over: the drop rate, or the weight decay
    runs["value"] = np.where(runs["method"] == "WD", runs["weight_decay"], runs["rate"])
    main = np.ones(len(runs), dtype=bool)
    for k, v in MAIN.items():
        main &= runs[k] == v
    runs["experiment"] = np.select(
        [~main, runs["runs_dir"] == "lora_runs_ckpt", runs["noise"] == "gaussian", runs["schedule"] != "uniform"],
        ["lora-hparams", "checkpoints", "gaussian", "schedules"], default="ladder")
    return runs


def load_curves():
    return pd.read_parquet(CURVES)


# ---------------------------------------------------------------------------
# Best-of-grid selection and paired statistics

def main_grid(runs):
    """The post's grid: L0, L1-L6 at rates <= 0.3 (uniform schedule, Bernoulli masks), and the weight-decay grid."""
    r = runs[runs["experiment"] == "ladder"]
    keep = (r["method"] == "L0") | (r["method"].isin(METHODS[1:7]) & r["rate"].isin(RATES)) | \
           ((r["method"] == "WD") & r["weight_decay"].isin(WEIGHT_DECAYS))
    return r[keep].reset_index(drop=True)


def best_of_grid(df, metric="pass_floor"):
    """One row per (base, dataset, seed, method): the grid value minimizing `metric`."""
    idx = df.groupby(["base", "dataset", "seed", "method"])[metric].idxmin()
    return df.loc[idx].reset_index(drop=True)


def paired(best, a, b, base, metric="pass_floor"):
    """Paired differences metric(a) - metric(b) over (dataset, seed) for one base."""
    piv = best[best["base"] == base].pivot_table(index=["dataset", "seed"], columns="method", values=metric)
    if a not in piv or b not in piv:
        return pd.Series(dtype=float)
    return (piv[a] - piv[b]).dropna()


def summarize(diffs, alternative="greater"):
    d = diffs.to_numpy()
    n = len(d)
    w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    mean, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    p = stats.wilcoxon(d, alternative=alternative).pvalue if np.any(d != 0) else 1.0
    return dict(n=n, wins=w, ties=t, losses=l, mean=mean, ci_lo=mean - half, ci_hi=mean + half, p=p)


def relative(best, a, b, base, metric="pass_floor"):
    """Mean over pairs of (a - b) / b, the relative difference the post reports next to nats."""
    piv = best[best["base"] == base].pivot_table(index=["dataset", "seed"], columns="method", values=metric)
    return float(((piv[a] - piv[b]) / piv[b]).dropna().mean())


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.4f}"


# ---------------------------------------------------------------------------
# Report

def report_headline(best):
    print("\n=== Every method vs DLP (L5), best grid value per dataset and seed (Figure 5, Appendix C) ===")
    print("gap = method - DLP in nats (positive: DLP better); record = DLP wins-ties-losses over 15 pairs")
    for base in BASES:
        print(f"\n  {base} base")
        for m in METHODS:
            if m == DLP:
                continue
            d = paired(best, m, DLP, base)
            if d.empty:
                continue
            s = summarize(d)
            rel = relative(best, m, DLP, base)
            print(f"  {m:3s} {METHOD_NAMES[m]:<22} {s['wins']:>2}-{s['ties']}-{s['losses']:<2} gap {s['mean']:.4f} "
                  f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] ({100 * rel:.1f}%, {100 * (1 - np.exp(-s['mean'])):.1f}% in "
                  f"perplexity)  p={fmt_p(s['p'])}")


def report_absolute(best):
    print("\n=== Best-of-grid validation loss, mean ± sd over seeds (Appendix C) ===")
    tab = best.groupby(["base", "dataset", "method"])["pass_floor"].agg(["mean", "std"])
    for base in BASES:
        print(f"\n  {base} base" + "".join(f"{ds:>20}" for ds in DATASETS))
        for m in METHODS:
            cells = [f"{tab.loc[(base, ds, m), 'mean']:.3f} ± {tab.loc[(base, ds, m), 'std']:.3f}"
                     if (base, ds, m) in tab.index else "-" for ds in DATASETS]
            print(f"  {m:3s} {METHOD_NAMES[m]:<22}" + "".join(f"{c:>20}" for c in cells))


def report_best_values(best):
    print("\n=== Chosen grid value per method (counts over dataset x seed) ===")
    for base in BASES:
        for m in METHODS[1:]:
            vc = best[(best["base"] == base) & (best["method"] == m)]["value"].value_counts().sort_index()
            if len(vc):
                print(f"  {base:9s} {m:3s} " + ", ".join(f"{k:g} x{v}" for k, v in vc.items()))


def report_rates(grid):
    print("\n=== DLP vs standard dropout at each rate, and averaged over rates (Figure 6) ===")
    for base in BASES:
        g = grid[(grid["base"] == base) & grid["method"].isin(["L1", DLP])]
        out = []
        for p in RATES:
            piv = g[g["rate"] == p].pivot_table(index=["dataset", "seed"], columns="method", values="pass_floor")
            if {"L1", DLP} <= set(piv):
                out.append(f"p={p:g}: {int(((piv['L1'] - piv[DLP]) > 0).sum())}/{len(piv)}")
        m = g.groupby(["dataset", "seed", "method"])["pass_floor"].mean().unstack("method")
        if {"L1", DLP} <= set(m):
            s = summarize(m["L1"] - m[DLP])
            out.append(f"mean over rates: {s['wins']}/{s['n']}, p={fmt_p(s['p'])}")
        print(f"  {base:9s} DLP better in " + "; ".join(out))
    print("\n=== Gain over no regularization, as a share of the unregularized loss (Discussion) ===")
    for base in BASES:
        g = grid[grid["base"] == base]
        l0 = g[g["method"] == "L0"].set_index(["dataset", "seed"])["pass_floor"]
        for m in ["L1", DLP]:
            parts = []
            for p in RATES:
                x = g[(g["method"] == m) & (g["rate"] == p)].set_index(["dataset", "seed"])["pass_floor"]
                rel = ((l0 - x) / l0).dropna()
                if len(rel):
                    parts.append(f"p={p:g}: {100 * rel.mean():.1f}%")
            print(f"  {base:9s} {m}: " + ", ".join(parts))


def report_droppath(best):
    print("\n=== L6 needs a drop-path base (L6 section) ===")
    gaps = {base: paired(best, "L6", DLP, base) for base in BASES}
    if all(len(g) for g in gaps.values()):
        diff = (gaps["standard"] - gaps["droppath"]).dropna()
        s = summarize(diff, alternative="two-sided")
        print(f"  L6 - DLP: standard {gaps['standard'].mean():.4f}, droppath {gaps['droppath'].mean():.4f}; "
              f"shrinkage {s['mean']:.4f} [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}], two-sided p={fmt_p(s['p'])}")
    top = best.groupby(["base", "dataset", "seed"])["pass_floor"].min().unstack("base")
    if set(BASES) <= set(top):
        d = (top["droppath"] - top["standard"]).dropna()
        rel = (d / top["standard"]).dropna().mean()
        print(f"  best result per dataset and seed, droppath - standard: {d.mean():+.4f} ({100 * rel:+.1f}%)")


def report_schedules(runs):
    print("\n=== Appendix D: DLP rate schedules (standard base, seeds 0-2, best peak rate in {0.1, 0.2, 0.3}) ===")
    sch = runs[(runs["base"] == "standard") & (runs["method"] == DLP) & (runs["noise"] == "bernoulli")
               & runs["rate"].isin([0.1, 0.2, 0.3]) & runs["seed"].isin([0, 1, 2])
               & runs["experiment"].isin(["ladder", "schedules"])]
    if (sch["schedule"] != "uniform").sum() == 0:
        print("  no schedule runs")
        return
    best = sch.loc[sch.groupby(["dataset", "seed", "schedule"])["pass_floor"].idxmin()]
    piv = best.pivot_table(index=["dataset", "seed"], columns="schedule", values="pass_floor")
    for other in ["linear_asc", "linear_desc"]:
        if other in piv:
            d = (piv[other] - piv["uniform"]).dropna()
            print(f"  {other} - uniform: {d.mean():+.4f} (uniform better in {(d > 0).sum()}/{len(d)})")
    if {"linear_asc", "linear_desc"} <= set(piv):
        d = (piv["linear_desc"] - piv["linear_asc"]).dropna()
        rel = (d / piv["linear_asc"]).dropna().mean()
        print(f"  linear_desc - linear_asc: {d.mean():+.4f} ({100 * rel:.1f}%), ascending better in "
              f"{(d > 0).sum()}/{len(d)}")


def report_gaussian(runs):
    print("\n=== Appendix G: Gaussian DLP vs each method at p = 0.3 (standard base) ===")
    g = runs[(runs["base"] == "standard") & (runs["rate"] == 0.3) & (runs["schedule"] == "uniform")
             & runs["experiment"].isin(["ladder", "gaussian"])].copy()
    g["method"] = np.where(g["noise"] == "gaussian", "L5-gaussian", g["method"])
    piv = g.pivot_table(index=["dataset", "seed"], columns="method", values="pass_floor")
    if "L5-gaussian" not in piv:
        print("  no Gaussian DLP runs")
        return
    for m in ["L1", "L2", "L3", "L4", "L5"]:
        if m in piv:
            d = (piv[m] - piv["L5-gaussian"]).dropna()
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            print(f"  {m} - Gaussian DLP: {d.mean():.4f}, Gaussian better in {(d > 0).sum()}/{len(d)}, "
                  f"two-sided p={fmt_p(p)}")


def report_lora_hparams(runs):
    print("\n=== Appendix B: LoRA sweep without regularization (OpenCodeReasoning, standard base, 3,000 steps) ===")
    s = runs[runs["experiment"] == "lora-hparams"].copy()
    if s.empty:
        print("  no sweep runs")
        return
    s["placement"] = np.where(s["lora_targets"] == "c_q,c_v", "q,v only", "all linear maps")
    best = s.loc[s.groupby(["placement", "rank", "effective_batch_size"])["pass_floor"].idxmin()]
    print(best.pivot_table(index=["placement", "rank"], columns="effective_batch_size",
                           values="pass_floor").round(3).to_string())
    top = s.loc[s["pass_floor"].idxmin()]
    print(f"  best: {top['placement']}, rank {top['rank']}, lr {top['lr']:g}, batch {top['effective_batch_size']}: "
          f"{top['pass_floor']:.3f}")


def report():
    runs = load()
    print(f"{len(runs)} runs: " + ", ".join(f"{k} {v}" for k, v in runs["experiment"].value_counts().items()))
    grid = main_grid(runs)
    best = best_of_grid(grid)
    report_headline(best)
    report_absolute(best)
    report_best_values(best)
    report_rates(grid)
    report_droppath(best)
    report_schedules(runs)
    report_gaussian(runs)
    report_lora_hparams(runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["collect", "report"])
    args = ap.parse_args()
    if args.command == "collect":
        collect()
    report()


if __name__ == "__main__":
    main()
