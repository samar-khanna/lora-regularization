"""Pair terms of the noise penalty over training (Appendix F, Figure F1; GPU).

The rise in training loss when a rung's noise is switched on at p = 0.3 estimates that rung's penalty Omega (the
post's Equation 1). The pair terms are Omega_L5 - Omega_L3: the rise under L5's noise (one mask value per sequence)
minus the rise under L3's (one per entry). Both are measured as in battery.py (64 training examples of the run's own
split, 2 mask draws), at steps 250, 500, 1000, 1500, ..., 4000 and 4499 of the checkpointed reruns of standard dropout
(L1) and DLP (L5) from `python -m scripts.sweep checkpoints`. Existing output files are skipped.

usage: python -m scripts.interpretability.pair_terms [--datasets gsm8k ...] [--seeds 0 1 ...] [--out DIR]
    -> <out>/pair_terms/<run id>/step<step>.json
"""
import argparse
import json
from pathlib import Path

import torch

from scripts.interpretability.battery import Battery, rel
from scripts.interpretability.common import DATASETS, OUT, SEEDS, ckpt_dir

STEPS = (250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4499)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out", type=Path, default=OUT, help="output root (default results/interpretability)")
    a = ap.parse_args()
    for ds in a.datasets:
        for seed in a.seeds:
            bat = None
            for rung in ("L1", "L5"):
                rdir = ckpt_dir(ds, rung, seed)
                for step in STEPS:
                    f = a.out / "pair_terms" / rdir.name / f"step{step:05d}.json"
                    if f.exists():
                        continue
                    path = rdir / ("adapters.pt" if step == 4499 else f"ckpt/adapters_{step:05d}.pt")
                    if not path.exists():
                        print(f"missing {path}", flush=True)
                        continue
                    if bat is None:
                        bat = Battery(ds, seed)
                    bat.load(path)
                    rec = dict(step=step, path=rel(path), noise=bat.noise(("L3", "L5")))
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(rec, indent=1))
                    n = rec["noise"]
                    print(f"{rdir.name} step {step}: pair terms {n['L5'] - n['L3']:+.4f}", flush=True)
            if bat is not None:
                del bat
                torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
