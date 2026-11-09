"""Loaders for the records in results/interpretability/, shared by report.py and
scripts/visualization/interpretability.py."""
import json

import numpy as np

from scripts.interpretability.common import FINAL_SET, OUT, tag
from scripts.results import load

FINAL = OUT / "final"
CKPT = OUT / "ckpt"
PAIR_TERMS = OUT / "pair_terms"
HYBRIDS = OUT / "hybrids"
METHODS = [m for m, _ in FINAL_SET]  # L0, WD, L1-L6
RATE = dict(FINAL_SET)  # each method's rate (weight decay for WD) in the records
# checkpointed reruns (python -m scripts.sweep checkpoints)
CKPT_RUN = {"L1": "{ds}_standard_L1_p0.3_s{seed}", "L5": "{ds}_standard_L5_p0.3_s{seed}"}
PAIR_RUNGS = ("L1", "L2", "L3", "L4", "L5")


def mean_se(a):
    a = np.asarray(a, dtype=float)
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))


def final_records(dataset):
    """{method: [battery record per seed]} for the final adapters (battery.py final)."""
    out = {}
    d = FINAL / dataset
    for r in METHODS:
        files = sorted(d.glob(f"{tag(r, RATE[r])}_s[0-9].json")) if d.exists() else []
        if files:
            out[r] = [json.loads(f.read_text()) for f in files]
    return out


def ckpt_curve(rung, dataset="gsm8k"):
    """Per-step means over complete seeds: steps, top-1, KL, validation loss, and the number of seeds."""
    per_seed = []
    for seed in range(5):
        d = CKPT / CKPT_RUN[rung].format(ds=dataset, seed=seed)
        files = sorted(d.glob("step*.json")) if d.exists() else []
        if len(files) < 18:
            continue
        recs = [json.loads(f.read_text()) for f in files]
        per_seed.append({r["step"]: (r["top1"], r["pre_kl"], r["val"]) for r in recs})
    if not per_seed:
        return None
    steps = sorted(set.intersection(*[set(d) for d in per_seed]))
    arr = np.array([[d[s] for s in steps] for d in per_seed])  # seeds x steps x 3
    return np.array(steps), arr.mean(0), len(per_seed)


def pair_term_curve(rung, dataset):
    """Steps and the pair terms Omega_L5 - Omega_L3 at each step, seeds x steps (pair_terms.py)."""
    per_seed = []
    for seed in range(5):
        d = PAIR_TERMS / CKPT_RUN[rung].format(ds=dataset, seed=seed)
        recs = [json.loads(f.read_text()) for f in sorted(d.glob("step*.json"))]
        per_seed.append({q["step"]: q["noise"]["L5"] - q["noise"]["L3"] for q in recs})
    steps = sorted(set.intersection(*[set(x) for x in per_seed]))
    return steps, np.array([[x[t] for t in steps] for x in per_seed])


def hybrid_records(dataset):
    """{method: [{a, b, ref, hybrid}, ...]} for one dataset (hybrids.py)."""
    return json.loads((HYBRIDS / f"{dataset}.json").read_text())


def pair_vs_val_data(datasets=("opencode", "gsm8k", "glaive")):
    runs = load()
    u = runs[(runs.base == "standard") & (runs.experiment == "ladder") & (runs.rate == 0.3)]
    out = {}
    for ds in datasets:
        piv = u[u.dataset == ds].pivot_table(index="seed", columns="method", values="pass_floor")
        rec = final_records(ds)
        for r in PAIR_RUNGS:
            gap = (piv[r] - piv["L5"]).dropna().to_numpy()
            per = np.array([q["noise"]["L3"] for q in rec[r]])
            tok = np.array([q["noise"]["L4"] for q in rec[r]])
            full = np.array([q["noise"]["L5"] for q in rec[r]])
            pair = full - per
            out[(ds, r)] = dict(gap=mean_se(gap), per=mean_se(per), tok=mean_se(tok), full=mean_se(full),
                                pair=mean_se(pair))
    return out
