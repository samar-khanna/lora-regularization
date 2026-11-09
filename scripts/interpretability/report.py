"""The numbers quoted in the post's "Why does Drop-LoRA-Path work?" section and Appendices E-F, from
results/interpretability/ (and results/runs.csv for Figure 12).

usage: python -m scripts.interpretability.report
"""
import numpy as np

from scripts.interpretability.common import DATASETS, tag
from scripts.interpretability.records import (METHODS, RATE, ckpt_curve, final_records, hybrid_records,
                                              pair_term_curve, pair_vs_val_data)

NAME = {"opencode": "OpenCodeReasoning", "gsm8k": "GSM8K", "glaive": "Glaive"}


def forgetting(rec):
    print("Figure 8: KL(base || fine-tuned) on pretraining text, nats per token (mean of 5 seeds)")
    kl = {ds: {r: np.mean([q["pre_kl"] for q in rec[ds][r]]) for r in METHODS} for ds in DATASETS}
    for ds in DATASETS:
        print(f"  {NAME[ds]:18s}" + "  ".join(f"{r} {kl[ds][r]:.2f}" for r in METHODS))
    m = {r: np.mean([kl[ds][r] for ds in DATASETS]) for r in METHODS}
    print(f"  mean over datasets: DLP (L5) {m['L5']:.2f}, {1 - m['L5'] / m['L1']:.0%} less than standard dropout (L1) "
          f"{m['L1']:.2f} and {1 - m['L5'] / m['L0']:.0%} less than no regularization {m['L0']:.2f}")


def forgetting_over_training():
    print("Figure 9: next-token accuracy (%) / KL at steps 500, 1000, ..., 4000 (mean of 5 seeds)")
    for ds in DATASETS:
        for r in ("L1", "L5"):
            steps, m, _ = ckpt_curve(r, ds)
            pts = "  ".join(f"{100 * acc:.1f}/{kl:.2f}" for s, (acc, kl, _) in zip(steps, m) if s % 500 == 0)
            print(f"  {NAME[ds]:18s}{r}: {pts}; lowest validation loss at step {steps[int(np.argmin(m[:, 2]))]}")


def coadaptation(rec):
    print("Figure 10: co-adaptation (validation loss on 64 examples)")
    for ds in DATASETS:
        alone = np.mean([np.array(q["only_val"]) - q["base_blk_val"] for q in rec[ds]["L0"]], 0)
        print(f"  {NAME[ds]:18s}no regularization: the weakest block alone is block {int(np.argmax(alone))}")
    print("  removing one block's adapters, change in validation loss (mean of 5 seeds):")
    for ds in ("gsm8k", "glaive"):
        for r in ("L0", "L1", "L5"):
            d = np.mean([np.array(q["knock_val"]) - q["full_val"] for q in rec[ds][r]], 0)
            rel = np.mean([(np.array(q["knock_val"]) - q["full_val"]) / q["full_val"] for q in rec[ds][r]], 0)
            print(f"    {NAME[ds]:16s}{r}: block 10 {d[10]:+.3f} ({rel[10]:+.1%}), "
                  f"block 11 {d[11]:+.3f} ({rel[11]:+.1%})")
    for r in ("L5", "L1"):
        d = np.mean([[q["base_blk_val"] - np.array(q["only_val"]) for q in rec[ds][r]] for ds in DATASETS])
        rel = np.mean([[(q["base_blk_val"] - np.array(q["only_val"])) / q["base_blk_val"] for q in rec[ds][r]]
                       for ds in DATASETS])
        print(f"  {r}: one block's adapters alone lower validation loss by {d:.2f} nats ({rel:.1%}) on average")


def penalties():
    print("Figure 12: Pearson r with validation loss minus DLP's, over 15 points (and the 12 without DLP's)")
    d = pair_vs_val_data(DATASETS)
    gap = np.array([v["gap"][0] for v in d.values()])
    other = np.array([r != "L5" for _, r in d])
    for key, name in (("per", "(a) Omega_L3"), ("tok", "(b) Omega_L4"), ("full", "(c) Omega_L5"),
                      ("pair", "(d) Omega_L5 - Omega_L3")):
        x = np.array([v[key][0] for v in d.values()])
        print(f"  {name:24s} r = {np.corrcoef(x, gap)[0, 1]:.2f} ({np.corrcoef(x[other], gap[other])[0, 1]:.2f})")
    print("  DLP's pair terms: " + ", ".join(f"{NAME[ds]} {d[(ds, 'L5')]['pair'][0]:+.3f}" for ds in DATASETS))


def hybrids():
    print("Figure E1: hybrid's validation loss minus the originals' (mean of 5 seed pairs)")
    for ds in DATASETS:
        h = hybrid_records(ds)
        parts = []
        for r in ("L0", "L1", "L5"):
            rows = h[tag(r, RATE[r])]
            g = np.mean([q["hybrid"] - q["ref"] for q in rows])
            rel = np.mean([(q["hybrid"] - q["ref"]) / q["ref"] for q in rows])
            parts.append(f"{r} {g:.3f} ({rel:.0%})")
        print(f"  {NAME[ds]:18s}" + ", ".join(parts))
    n = [q["n_problems"] for q in hybrid_records("opencode")[tag("L5", RATE["L5"])]]
    print(f"  OpenCodeReasoning: each pair is scored on the {min(n)}-{max(n)} problems both runs held out")


def pair_terms_over_training():
    print("Figure F1: pair terms Omega_L5 - Omega_L3 over training (mean of 5 seeds)")
    for ds in DATASETS:
        for r in ("L1", "L5"):
            steps, v = pair_term_curve(r, ds)
            print(f"  {NAME[ds]:18s}{r}: " + "  ".join(f"{s} {x:+.3f}" for s, x in zip(steps, v.mean(0))))


def main():
    rec = {ds: final_records(ds) for ds in DATASETS}
    for section in (lambda: forgetting(rec), forgetting_over_training, lambda: coadaptation(rec), penalties, hybrids,
                    pair_terms_over_training):
        section()
        print()


if __name__ == "__main__":
    main()
