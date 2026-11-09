"""Measurements on trained adapters for "Why does Drop-LoRA-Path work?" (GPU; read-only on the runs it measures).

final <dataset>: the final adapters of each method in FINAL_SET (common.py), seeds 0-4, standard base.
    pre_kl        KL(base || fine-tuned) per token on 32 x 1024 tokens of held-out pretraining text (Figure 8)
    base_blk_val  the base model's loss on the first 64 validation examples
    full_val      the fine-tuned model's loss on those examples
    knock_val     the same with one block's 6 adapters removed, for each of the 12 blocks
    only_val      the same with only one block's adapters kept, for each block (Figure 10)
    noise         rise in loss on 64 training examples when L3's, L4's or L5's noise is switched on in every adapter
                  at p = 0.3, mean of 2 mask draws: the penalties Omega of Figure 12
  -> <out>/final/<dataset>/<method>_p<rate>_s<seed>.json

ckpt <run id>: every checkpoint of a rerun from `python -m scripts.sweep checkpoints`.
    val, top1     loss and next-token accuracy on the validation set (GSM8K: all of it; OpenCodeReasoning and Glaive:
                  the first 512 examples)
    pre_kl        as above (Figure 9)
  -> <out>/ckpt/<run id>/step<step>.json

Existing output files are skipped.

usage:
  python -m scripts.interpretability.battery final <dataset> [--seeds 0 1 2 3 4] [--methods L1 L5] [--out DIR]
  python -m scripts.interpretability.battery ckpt <run id under lora_runs_ckpt> [--out DIR]
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from nanochat.common import get_base_dir
from nanochat.lora import ADAPTER_RUNGS
from nanochat.sft_data import PROMPT_MASK_VALUE
from scripts.interpretability.common import (BUILD, FINAL_SET, OUT, dev, load_adapters, load_lora_state_dict,
                                             lora_model, run_dir, tag)

N_VAL_BATCHES = {"gsm8k": 165, "glaive": 64, "opencode": 64}  # validation batches of 8 for `ckpt` (all of GSM8K)
N_BLK = 8  # batches of 8 for the block test (validation) and the penalties (training): 64 examples each


def rel(path):
    """A path relative to the nanochat base dir, for the records."""
    path = Path(path)
    try:
        return str(path.relative_to(get_base_dir()))
    except ValueError:
        return str(path)


def pretraining_batches(tok, n_seq=32, T=1024):
    """n_seq sequences of T tokens from the FineWeb-Edu validation shard, in batches of 4."""
    path = sorted((Path(get_base_dir()) / "base_data").glob("shard_*.parquet"))[-1]  # the validation shard
    pf = pq.ParquetFile(path)
    bos = tok.get_bos_token_id()
    ids, rg = [], 0
    while len(ids) < n_seq * (T + 1):
        for t in pf.read_row_group(rg).column("text").to_pylist():
            ids += tok.encode(t, prepend=bos)
            if len(ids) >= n_seq * (T + 1):
                break
        rg += 1
    a = torch.tensor(ids[: n_seq * (T + 1)], device=dev).view(n_seq, T + 1)
    return [(a[i:i + 4, :-1].contiguous(), a[i:i + 4, 1:].contiguous()) for i in range(0, n_seq, 4)]


class Battery:
    """The standard base model with LoRA on every linear map, one run's data (its dataset and seed), and the base
    model's reference values. Load a set of adapters with `load`, then measure."""

    def __init__(self, dataset, seed):
        self.model, self.tok, self.mods = lora_model()
        self.NL = len(self.model.transformer.h)
        tr_it, va_it = BUILD[dataset](self.tok, 2048, 8, dev, seed=seed)
        self.val = [next(va_it)]  # one full cycle of the (unshuffled, cycling) validation iterator
        while True:
            b = next(va_it)
            if b[0].shape == self.val[0][0].shape and torch.equal(b[0], self.val[0][0]):
                break
            self.val.append(b)
        self.val = self.val[:N_VAL_BATCHES[dataset]]
        self.train = [next(tr_it) for _ in range(N_BLK)]
        self.pre = pretraining_batches(self.tok)
        self.on(set())  # the base model
        with torch.no_grad():
            self.base_pre_logp = [F.log_softmax(self.model(x).float(), -1).to(torch.bfloat16) for x, _ in self.pre]
        self.base_blk_val = self.mean_loss(self.val[:N_BLK])

    # ------------------------------------------------------------------ adapter control
    def load(self, path):
        load_lora_state_dict(self.model, load_adapters(path), strict=True)
        self.on(set(range(self.NL)))

    def on(self, blocks):
        """Keep the adapters of `blocks`, noise-free, and remove the others."""
        for name, m in self.mods:
            m.scale = (m.alpha / m.rank) if int(name.split(".")[2]) in blocks else 0.0
            m.rung, m.rate, m.noise = "L0", 0.0, "bernoulli"
            m.eval()

    # ------------------------------------------------------------------ measurements
    @torch.no_grad()
    def logits_sup(self, x, y):
        """Logits and targets at the supervised (response) positions."""
        logits = self.model(x)
        m = y != PROMPT_MASK_VALUE
        return logits[m].float(), y[m]

    @torch.no_grad()
    def mean_loss(self, batches):
        tot, n = 0.0, 0
        for x, y in batches:
            lg, t = self.logits_sup(x, y)
            tot += F.cross_entropy(lg, t, reduction="sum").item()
            n += len(t)
        return tot / n

    @torch.no_grad()
    def val_top1(self):
        """Loss and next-token accuracy on the validation batches."""
        tot, ok, n = 0.0, 0, 0
        for x, y in self.val:
            lg, t = self.logits_sup(x, y)
            tot += F.cross_entropy(lg, t, reduction="sum").item()
            ok += int((lg.argmax(-1) == t).sum())
            n += len(t)
        return tot / n, ok / n

    @torch.no_grad()
    def pre_kl(self):
        """KL(base || fine-tuned) per token on the pretraining text."""
        kl, n = 0.0, 0
        for (x, _), blp in zip(self.pre, self.base_pre_logp):
            lp = F.log_softmax(self.model(x).float(), -1)
            b = blp.float()
            kl += (b.exp() * (b - lp)).sum().item()
            n += x.numel()
        return kl / n

    def blocks_test(self):
        """Validation loss with every adapter, without one block's adapters, and with only one block's."""
        bv, every = self.val[:N_BLK], set(range(self.NL))
        r = dict(full_val=self.mean_loss(bv), knock_val=[], only_val=[])
        for b in range(self.NL):
            self.on(every - {b})
            r["knock_val"].append(self.mean_loss(bv))
            self.on({b})
            r["only_val"].append(self.mean_loss(bv))
        self.on(every)
        return r

    @torch.no_grad()
    def noise(self, rungs=("L3", "L4", "L5"), p=0.3, draws=2):
        """Rise in training loss when a rung's noise is switched on in every adapter, as during training: the
        penalty Omega of the post's Equation 1, summed over adapters. Mean of `draws` mask draws, each seeded by the
        draw and the rung."""
        clean = self.mean_loss(self.train)
        out = {}
        for rung in rungs:
            rises = []
            for d in range(draws):
                torch.manual_seed(1000 * d + ADAPTER_RUNGS.index(rung))
                for _, m in self.mods:
                    m.rung, m.rate = rung, p
                    m.train()
                rises.append(self.mean_loss(self.train) - clean)
                self.on(set(range(self.NL)))
            out[rung] = sum(rises) / draws
        return out

    # ------------------------------------------------------------------ records
    def measure_final(self, path):
        self.load(path)
        rec = dict(path=rel(path), pre_kl=self.pre_kl(), base_blk_val=self.base_blk_val)
        rec.update(self.blocks_test())
        rec["noise"] = self.noise()
        return rec

    def measure_ckpt(self, path, step):
        self.load(path)
        val, top1 = self.val_top1()
        return dict(step=step, path=rel(path), val=val, top1=top1, pre_kl=self.pre_kl())


def write(rec, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["final", "ckpt"])
    ap.add_argument("target", help="dataset (final) or run id under lora_runs_ckpt (ckpt)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="final only")
    ap.add_argument("--methods", nargs="+", default=None, help="final only: limit FINAL_SET, e.g. L5 L1 WD")
    ap.add_argument("--out", type=Path, default=OUT, help="output root (default results/interpretability)")
    a = ap.parse_args()
    if a.mode == "final":
        for seed in a.seeds:
            todo = [(m, p, a.out / "final" / a.target / f"{tag(m, p)}_s{seed}.json") for m, p in FINAL_SET
                    if a.methods is None or m in a.methods]
            todo = [t for t in todo if not t[2].exists()]
            if not todo:
                continue
            bat = Battery(a.target, seed)
            for m, p, f in todo:
                rec = bat.measure_final(run_dir(a.target, m, p, seed) / "adapters.pt")
                write(rec, f)
                print(f"{a.target} {f.stem}: pre_kl {rec['pre_kl']:.3f} full_val {rec['full_val']:.4f} "
                      + " ".join(f"{k} {v:.4f}" for k, v in rec["noise"].items()), flush=True)
            del bat
            torch.cuda.empty_cache()
    else:
        rd = Path(get_base_dir()) / "lora_runs_ckpt" / a.target
        cfg = json.loads((rd / "summary.json").read_text())["config"]
        ckpts = [(int(p.stem.split("_")[1]), p) for p in sorted((rd / "ckpt").glob("adapters_*.pt"))]
        ckpts.append((cfg["num_iterations"] - 1, rd / "adapters.pt"))
        todo = [(s, p, a.out / "ckpt" / a.target / f"step{s:05d}.json") for s, p in ckpts]
        todo = [t for t in todo if not t[2].exists()]
        if todo:
            bat = Battery(cfg["dataset"], cfg["seed"])
            for step, path, f in todo:
                rec = bat.measure_ckpt(path, step)
                write(rec, f)
                print(f"{a.target} step {step}: val {rec['val']:.4f} top1 {rec['top1']:.4f} "
                      f"pre_kl {rec['pre_kl']:.3f}", flush=True)


if __name__ == "__main__":
    main()
