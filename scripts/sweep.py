"""Launch the fine-tuning runs behind the post.

Each run is one call of `scripts.lora_sft`, logged to <run dir>/stdout.log. Runs whose summary.json already exists
are skipped, so a sweep can be stopped and restarted. Several runs can share one GPU (--parallel); the post used
2-5 at a time on a single GB10 with the CUDA MPS daemon running.

Presets (the defaults for --datasets, --bases and --seeds are the post's, and can be narrowed):

    ladder        L0, plus L1-L6 at p in {0.05, 0.1, 0.2, 0.3}          Figures 4-7, Appendix C
    weight-decay  L0 with AdamW weight decay in {0.01, ..., 0.4}         Figures 4-5, Appendix C
    schedules     L5 with depth schedules linear_asc / linear_desc,      Appendix D
                  peak p in {0.1, 0.2, 0.3}; standard base, seeds 0-2
    gaussian      L5 with Gaussian noise ("Gaussian DLP") at p = 0.3;     Appendix G
                  standard base
    checkpoints   L1 and L5 at p = 0.3, adapters saved every 250 steps;   Figure 9, Appendix F (and the
                  standard base; written to <base>/lora_runs_ckpt/        interpretability analyses on checkpoints)
    lora-hparams  no regularization: placement x rank x learning rate     Appendix B
                  x batch size, 3,000 steps; OpenCodeReasoning,
                  standard base, seed 0

With 3 datasets, 2 bases and 5 seeds, `ladder` is 30 + 720 runs and `weight-decay` 180, the 930 runs of the post.

Usage:
    python -m scripts.sweep ladder --dry-run
    python -m scripts.sweep ladder --parallel 2
    python -m scripts.sweep ladder --datasets gsm8k --bases standard --seeds 0 --rungs L1,L5 --rates 0.3
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nanochat.common import get_base_dir

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = ["opencode", "gsm8k", "glaive"]
BASES = ["standard", "droppath"]  # model tags written by runs/pretrain.sh
SEEDS = [0, 1, 2, 3, 4]
RATES = [0.05, 0.1, 0.2, 0.3]
WEIGHT_DECAYS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
ALL_LINEAR = "c_q,c_k,c_v,c_proj,c_fc"

# The LoRA configuration chosen by the lora-hparams sweep, fixed for everything else.
FIXED = {
    "lora_targets": ALL_LINEAR,
    "rank": 32,
    "alpha": 32.0,
    "lr": 3e-3,
    "effective_batch_size": 16,
    "device_batch_size": 8,
    "max_seq_len": 2048,
    "num_iterations": 4500,
    "warmup_steps": 50,
    "final_lr_frac": 0.0,
    "grad_clip": 1.0,
    "eval_every": 50,
    "eval_batches": 20,
    "weight_decay": 0.0,
    "rung": "L0",
    "rate": 0.0,
    "noise": "bernoulli",
    "schedule": "uniform",
    "save_every": 0,
    "runs_dir": "lora_runs",
}


def run_id(c):
    if c.get("preset") == "lora-hparams":
        place = "qv" if c["lora_targets"] == "c_q,c_v" else "all"
        return f"lorahp_{c['dataset']}_{c['base']}_{place}_r{c['rank']}_lr{c['lr']:g}_b{c['effective_batch_size']}_s{c['seed']}"
    rid = f"{c['dataset']}_{c['base']}_{c['rung']}"
    if c["rung"] != "L0":
        rid += f"_p{c['rate']:g}"
    if c["weight_decay"] > 0:
        rid += f"_wd{c['weight_decay']:g}"
    if c["schedule"] != "uniform":
        rid += "_" + {"linear_asc": "asc", "linear_desc": "desc"}[c["schedule"]]
    if c["noise"] != "bernoulli":
        rid += f"_{c['noise']}"
    return rid + f"_s{c['seed']}"


def cell(**kw):
    c = dict(FIXED)
    c.update(kw)
    return c


def build_cells(preset, datasets, bases, seeds, rungs=None, rates=None):
    cells = []
    if preset == "ladder":
        for ds, base, seed in itertools.product(datasets, bases, seeds):
            if rungs is None or "L0" in rungs:
                cells.append(cell(dataset=ds, base=base, seed=seed))
            for rung in [r for r in ("L1", "L2", "L3", "L4", "L5", "L6") if rungs is None or r in rungs]:
                for p in (rates or RATES):
                    cells.append(cell(dataset=ds, base=base, seed=seed, rung=rung, rate=p))
    elif preset == "weight-decay":
        for ds, base, seed in itertools.product(datasets, bases, seeds):
            for wd in (rates or WEIGHT_DECAYS):
                cells.append(cell(dataset=ds, base=base, seed=seed, weight_decay=wd))
    elif preset == "schedules":
        for ds, base, seed in itertools.product(datasets, bases, seeds):
            for sched in ("linear_asc", "linear_desc"):
                for p in (rates or [0.1, 0.2, 0.3]):
                    cells.append(cell(dataset=ds, base=base, seed=seed, rung="L5", rate=p, schedule=sched))
    elif preset == "gaussian":
        for ds, base, seed in itertools.product(datasets, bases, seeds):
            for p in (rates or [0.3]):
                cells.append(cell(dataset=ds, base=base, seed=seed, rung="L5", rate=p, noise="gaussian"))
    elif preset == "checkpoints":
        for ds, base, seed in itertools.product(datasets, bases, seeds):
            for rung in [r for r in ("L1", "L5") if rungs is None or r in rungs]:
                for p in (rates or [0.3]):
                    cells.append(cell(dataset=ds, base=base, seed=seed, rung=rung, rate=p,
                                      save_every=250, runs_dir="lora_runs_ckpt"))
    elif preset == "lora-hparams":
        for ds, base, seed in itertools.product(datasets, bases, seeds):
            for targets, rank, lr, eb in itertools.product(["c_q,c_v", ALL_LINEAR], [4, 8, 16, 32],
                                                           [3e-4, 1e-3, 3e-3, 1e-2], [16, 32, 64]):
                cells.append(cell(dataset=ds, base=base, seed=seed, preset="lora-hparams", lora_targets=targets,
                                  rank=rank, lr=lr, effective_batch_size=eb, num_iterations=3000))
    else:
        raise ValueError(preset)
    return cells


def command(c, rid, use_wandb):
    args = [sys.executable, "-m", "scripts.lora_sft",
            "--base-model-tag", c["base"], "--dataset", c["dataset"], "--seed", str(c["seed"]),
            "--lora-targets", c["lora_targets"], "--rank", str(c["rank"]), "--alpha", str(c["alpha"]),
            "--rung", c["rung"], "--rate", f"{c['rate']:g}", "--noise", c["noise"], "--schedule", c["schedule"],
            "--lr", f"{c['lr']:g}", "--weight-decay", f"{c['weight_decay']:g}",
            "--effective-batch-size", str(c["effective_batch_size"]),
            "--device-batch-size", str(c["device_batch_size"]), "--max-seq-len", str(c["max_seq_len"]),
            "--num-iterations", str(c["num_iterations"]), "--warmup-steps", str(c["warmup_steps"]),
            "--final-lr-frac", str(c["final_lr_frac"]), "--grad-clip", str(c["grad_clip"]),
            "--eval-every", str(c["eval_every"]), "--eval-batches", str(c["eval_batches"]),
            "--save-every", str(c["save_every"]), "--runs-dir", c["runs_dir"],
            "--run-id", rid, "--run", rid if use_wandb else "dummy"]
    return args


def run_cell(c, use_wandb):
    rid = run_id(c)
    out_dir = Path(get_base_dir()) / c["runs_dir"] / rid
    if (out_dir / "summary.json").exists():
        return rid, "skipped", None
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"
    env = dict(os.environ, NANOCHAT_BASE_DIR=get_base_dir(), OMP_NUM_THREADS="1")
    with open(log_path, "w") as f:
        proc = subprocess.run(command(c, rid, use_wandb), env=env, cwd=str(REPO_ROOT),
                              stdout=f, stderr=subprocess.STDOUT)
    return rid, ("ok" if proc.returncode == 0 else "failed"), str(log_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=["ladder", "weight-decay", "schedules", "gaussian", "checkpoints", "lora-hparams"])
    ap.add_argument("--datasets", type=str, default=None, help="comma-separated subset of opencode,gsm8k,glaive")
    ap.add_argument("--bases", type=str, default=None, help="comma-separated subset of standard,droppath")
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated seeds")
    ap.add_argument("--rungs", type=str, default=None, help="ladder/checkpoints: comma-separated rungs to keep")
    ap.add_argument("--rates", type=str, default=None,
                    help="comma-separated rates to use instead of the preset's (weight decays for weight-decay)")
    ap.add_argument("--parallel", type=int, default=2, help="runs at a time (default 2)")
    ap.add_argument("--wandb", action="store_true", help="log every run to wandb (project lora-regularization)")
    ap.add_argument("--dry-run", action="store_true", help="list the runs without launching them")
    args = ap.parse_args()

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None
    defaults = {  # (datasets, bases, seeds) per preset
        "schedules": (DATASETS, ["standard"], [0, 1, 2]),
        "gaussian": (DATASETS, ["standard"], SEEDS),
        "checkpoints": (DATASETS, ["standard"], SEEDS),
        "lora-hparams": (["opencode"], ["standard"], [0]),
    }.get(args.preset, (DATASETS, BASES, SEEDS))
    datasets = split(args.datasets) or defaults[0]
    bases = split(args.bases) or defaults[1]
    seeds = [int(s) for s in split(args.seeds)] if args.seeds else defaults[2]
    rates = [float(r) for r in split(args.rates)] if args.rates else None
    cells = build_cells(args.preset, datasets, bases, seeds, rungs=split(args.rungs), rates=rates)

    base = Path(get_base_dir())
    todo = [c for c in cells if not (base / c["runs_dir"] / run_id(c) / "summary.json").exists()]
    print(f"{args.preset}: {len(cells)} runs, {len(cells) - len(todo)} already done, {len(todo)} to run "
          f"(datasets {datasets}, bases {bases}, seeds {seeds}); outputs under {base}")
    if args.dry_run:
        for c in cells:
            print("  " + run_id(c))
        return
    if not todo:
        return

    t0 = time.time()
    status = {"ok": [], "failed": [], "skipped": []}
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(run_cell, c, args.wandb) for c in todo]
        for fut in as_completed(futures):
            rid, st, log = fut.result()
            status[st].append(rid)
            done = sum(len(v) for v in status.values())
            note = f"  ({log})" if st == "failed" else ""
            print(f"[{(time.time() - t0) / 60:6.1f}m] [{done}/{len(todo)}] {st:7s} {rid}{note}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min: {len(status['ok'])} ok, {len(status['failed'])} failed")
    if status["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
