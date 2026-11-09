"""Adapters swapped between training runs (Appendix E, Figure E1; GPU).

For each method, take the final adapters of two training runs that differ only in seed and build a hybrid from the
first run's adapters in blocks 0-5 and the second run's in blocks 6-11. Seed pairs (0,1), (1,2), (2,3), (3,4), (4,0).
Updates that only work with their own run's other blocks break when combined; self-contained ones don't.

GSM8K and Glaive: scored on the first 32 validation batches (their validation split does not depend on the seed).
OpenCodeReasoning: the split is drawn per seed, so each pair is scored on the problems both runs held out (the split
is rebuilt here with the same logic as nanochat.sft_data.build_opencode_sft_dataset).

Output: {method: [{a, b, ref, hybrid}, ...]}, where ref is the mean validation loss of the two original runs and
hybrid the hybrid's, on the same examples. OpenCodeReasoning's rows also give n_problems, the number of problems both
runs held out. An existing output file is not overwritten.

usage: python -m scripts.interpretability.hybrids <dataset> [--out DIR]   -> <out>/hybrids/<dataset>.json
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from nanochat.sft_data import PROMPT_MASK_VALUE, _prepare_opencode_cache
from scripts.interpretability.common import BUILD, FINAL_SET, OUT, dev, load_adapters, lora_model, run_dir, tag

SEEDS, PAIRS = [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]


def combine(mods, sd_a, sd_b, from_b):
    with torch.no_grad():
        for name, m in mods:
            src = sd_b if from_b(int(name.split(".")[2])) else sd_a
            m.lora_A.copy_(src[f"{name}.lora_A"])
            m.lora_B.copy_(src[f"{name}.lora_B"])
            m.scale = m.alpha / m.rank


@torch.no_grad()
def loss(model, val):
    tot, n = 0.0, 0
    for x, y in val:
        logits = model(x).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=PROMPT_MASK_VALUE,
                               reduction="sum").item()
        n += int((y != PROMPT_MASK_VALUE).sum())
    return tot / n


def opencode_pair_batches(tok):
    """For each seed pair, the number of problems both runs held out and their validation examples as padded
    batches."""
    table = pq.read_table(_prepare_opencode_cache(tok, 2048))
    tokens_list, plen_list, id_list = (table[c].to_pylist() for c in ("tokens", "prompt_length", "id"))
    by_id = defaultdict(list)
    for i, qid in enumerate(id_list):
        by_id[qid].append(i)
    unique_ids = sorted(by_id)

    def val_ids(seed):
        ids = list(unique_ids)
        random.Random(seed).shuffle(ids)
        return set(ids[:max(1, int(0.1 * len(ids)))])

    def batches(idx, bs=8):
        out = []
        for k in range(0, len(idx), bs):
            toks = [tokens_list[i] for i in idx[k:k + bs]]
            plens = [plen_list[i] for i in idx[k:k + bs]]
            T = max(len(t) for t in toks) - 1
            x = torch.zeros((len(toks), T), dtype=torch.long)
            y = torch.full((len(toks), T), PROMPT_MASK_VALUE, dtype=torch.long)
            for j, (t, pl) in enumerate(zip(toks, plens)):
                x[j, :len(t) - 1] = torch.tensor(t[:-1])
                if len(t) - 1 > pl - 1 and pl >= 1:
                    y[j, pl - 1:len(t) - 1] = torch.tensor(t[pl:])
            out.append((x.to(dev), y.to(dev)))
        return out

    val = {s: val_ids(s) for s in SEEDS}
    pair_data = {}
    for a, b in PAIRS:
        common = sorted(val[a] & val[b])
        idx = [i for q in common for i in by_id[q]]
        pair_data[(a, b)] = (len(common), batches(idx))
        print(f"pair {a}-{b}: {len(common)} problems held out by both, {len(idx)} examples", flush=True)
    return pair_data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["opencode", "gsm8k", "glaive"])
    ap.add_argument("--out", type=Path, default=OUT, help="output root (default results/interpretability)")
    args = ap.parse_args()
    ds = args.dataset
    out = args.out / "hybrids" / f"{ds}.json"
    if out.exists():
        print(f"{out} exists; pass --out to recompute", flush=True)
        return
    model, tok, mods = lora_model()
    if ds == "opencode":
        pair_data = opencode_pair_batches(tok)
    else:
        _, va_it = BUILD[ds](tok, 2048, 8, dev, seed=0)
        val = [next(va_it) for _ in range(32)]

    res = {}
    for method, p in FINAL_SET:
        sds = {s: load_adapters(run_dir(ds, method, p, s) / "adapters.pt") for s in SEEDS}
        rows = []
        if ds == "opencode":
            for a, b in PAIRS:
                n_problems, v = pair_data[(a, b)]
                combine(mods, sds[a], sds[a], lambda blk: False)
                loss_a = loss(model, v)
                combine(mods, sds[b], sds[b], lambda blk: False)
                loss_b = loss(model, v)
                combine(mods, sds[a], sds[b], lambda blk: blk >= 6)
                rows.append(dict(a=a, b=b, ref=0.5 * (loss_a + loss_b), hybrid=loss(model, v), n_problems=n_problems))
        else:
            single = {}
            for s in SEEDS:
                combine(mods, sds[s], sds[s], lambda blk: False)
                single[s] = loss(model, val)
            for a, b in PAIRS:
                combine(mods, sds[a], sds[b], lambda blk: blk >= 6)
                rows.append(dict(a=a, b=b, ref=0.5 * (single[a] + single[b]), hybrid=loss(model, val)))
        res[tag(method, p)] = rows
        gap = sum(r["hybrid"] - r["ref"] for r in rows) / len(rows)
        print(f"{tag(method, p)}: hybrid minus originals {gap:+.4f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
