"""LoRA fine-tuning with one rung of the regularization ladder.

Loads a pretrained base model, wraps the target linear maps with LoRA, and trains only the adapters with AdamW
on a linear-warmup + cosine schedule. The loss covers response tokens only. Every --eval-every steps it scores the
next slice of the validation set (--eval-batches micro-batches); see scripts/results.py for how those slices are
combined into the post's metric.

Rungs (--rung, numbered as in the post; --rate sets p):
  L0  no regularization
  L1  dropout(x)        dropout on the adapter's input: standard LoRA dropout (loralib, Hugging Face PEFT)
  L2  dropout(Ax)       dropout on the rank-r bottleneck
  L3  dropout(BAx)      dropout on the adapter's output, one mask value per element
  L4  token DLP         drop the adapter's output for whole tokens
  L5  sequence DLP      drop the adapter's output for whole sequences ("DLP")
  L6  drop-path         drop each attention / MLP sublayer (frozen weights and adapters together) for whole
                        sequences; the rate rises linearly with depth, from 0 in the first block to p in the last
--noise gaussian (L3-L5) keeps the rung's mask shape but draws a factor g ~ N(1, p/(1-p)) instead of keep/drop
(L5 + gaussian = the post's "Gaussian DLP", Appendix G). --schedule (L1-L5) spreads the rate across depth
(Appendix D). Weight decay is plain AdamW decay on A and B: --rung L0 --weight-decay 0.4.

Outputs go to <NANOCHAT_BASE_DIR>/<--runs-dir>/<--run-id>/: adapters.pt, summary.json (config, final metrics,
samples), train_log.parquet (train and eval rows) and lora_norms.parquet. --save-every N also writes
ckpt/adapters_<step>.pt at every eval step that is a multiple of N.

Example (the post's DLP setting on GSM8K):
  python -m scripts.lora_sft --base-model-tag standard --dataset gsm8k --rung L5 --rate 0.3 \
      --lora-targets c_q,c_k,c_v,c_proj,c_fc --rank 32 --alpha 32 --lr 3e-3 --effective-batch-size 16 \
      --num-iterations 4500 --seed 0
"""

import os
import json
import time
import math
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb,
    get_base_dir, autodetect_device_type, COMPUTE_DTYPE,
)
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_token_bytes
from nanochat.lora import apply_lora, get_lora_params, get_lora_state_dict, LoRALinear, RUNGS, RUNG_NAMES, NOISES
from nanochat.sft_data import (
    build_opencode_sft_dataset, build_gsm8k_sft_dataset, build_glaive_sft_dataset, PROMPT_MASK_VALUE,
)
from nanochat.engine import Engine


# -----------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="LoRA SFT training")
# Base model
parser.add_argument("--base-model-tag", type=str, required=True,
                    help="model tag of the base checkpoint (standard or droppath in the post)")
parser.add_argument("--base-model-step", type=int, default=None,
                    help="step of the base checkpoint to load (None = last)")
# LoRA architecture
parser.add_argument("--lora-targets", type=str, default="c_q,c_v",
                    help="comma-separated substrings; e.g. c_q,c_v or c_q,c_k,c_v,c_proj,c_fc")
parser.add_argument("--rank", type=int, default=8)
parser.add_argument("--alpha", type=float, default=32.0)
# Regularization (rungs numbered as in the post; see nanochat/lora.py)
parser.add_argument("--rung", type=str, default="L0", choices=list(RUNGS),
                    help="; ".join(f"{k}={v}" for k, v in RUNG_NAMES.items()))
parser.add_argument("--rate", type=float, default=0.0,
                    help="drop rate p (for L6: the rate at the deepest block, rising linearly from 0)")
parser.add_argument("--noise", type=str, default="bernoulli", choices=list(NOISES),
                    help="gaussian (L3-L5 only): factor g ~ N(1, p/(1-p)) with the rung's mask shape")
parser.add_argument("--schedule", type=str, default="uniform",
                    choices=["uniform", "linear_asc", "linear_desc"],
                    help="per-layer schedule of the adapter rate for L1-L5. "
                         "'uniform' (default, used throughout the post): every layer gets --rate. "
                         "'linear_asc': layer 0 -> 0, last layer -> rate (ViT/timm). "
                         "'linear_desc': layer 0 -> rate, last layer -> 0. "
                         "L6 always uses the linear-ascending drop-path schedule.")
# Optimization
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--effective-batch-size", type=int, default=32,
                    help="examples per optimizer step (= device-batch-size * grad-accum)")
parser.add_argument("--device-batch-size", type=int, default=8,
                    help="examples per forward/backward; reduce if OOM")
parser.add_argument("--max-seq-len", type=int, default=2048,
                    help="matches the base model's pretraining context length")
parser.add_argument("--num-iterations", type=int, default=500,
                    help="number of optimizer steps")
parser.add_argument("--warmup-steps", type=int, default=50)
parser.add_argument("--final-lr-frac", type=float, default=0.0,
                    help="final LR as fraction of peak (cosine endpoint)")
parser.add_argument("--grad-clip", type=float, default=1.0)
# Eval cadence
parser.add_argument("--eval-every", type=int, default=50,
                    help="run val eval every N optimizer steps")
parser.add_argument("--eval-batches", type=int, default=20,
                    help="number of val micro-batches per eval")
# Logging / output
parser.add_argument("--run", type=str, default="dummy",
                    help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--run-id", type=str, default=None,
                    help="output directory name; default uses dataset + model tag + rung + rate + seed + timestamp")
parser.add_argument("--runs-dir", type=str, default="lora_runs",
                    help="directory under the nanochat base dir for the run's outputs")
parser.add_argument("--save-every", type=int, default=0,
                    help="also save adapters every N steps (at eval steps; 0 = only the final adapters)")
parser.add_argument("--seed", type=int, default=0)
# Dataset
parser.add_argument("--dataset", type=str, default="opencode", choices=["opencode", "gsm8k", "glaive"],
                    help="SFT data: 'opencode' = OpenCodeReasoning (HackerEarth), "
                         "'gsm8k' = GSM8K math word problems, 'glaive' = Glaive function calling v2")
args = parser.parse_args()
if args.noise == "gaussian" and args.rung not in ("L3", "L4", "L5"):
    parser.error("--noise gaussian applies to rungs L3, L4 and L5")
if args.schedule != "uniform" and args.rung not in ("L1", "L2", "L3", "L4", "L5"):
    parser.error("--schedule applies to rungs L1-L5 (L6 always uses the linear depth schedule)")


# -----------------------------------------------------------------------------
# Init

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
torch.manual_seed(args.seed)

print0(f"Device: {device}, COMPUTE_DTYPE: {COMPUTE_DTYPE}")
print0(f"Rung: {args.rung} ({RUNG_NAMES[args.rung]}), rate: {args.rate}, noise: {args.noise}, "
       f"schedule: {args.schedule}, targets: {args.lora_targets}, rank: {args.rank}, alpha: {args.alpha}")

# Output directory
base_dir = get_base_dir()
runs_root = os.path.join(base_dir, args.runs_dir)
if args.run_id is None:
    args.run_id = (f"{args.dataset}_{args.base_model_tag}_{args.rung}_p{args.rate:g}_s{args.seed}_"
                   f"{time.strftime('%Y%m%d_%H%M%S')}")
out_dir = os.path.join(runs_root, args.run_id)
if master_process:
    os.makedirs(out_dir, exist_ok=True)
print0(f"Output directory: {out_dir}")

# wandb
use_dummy_wandb = args.run == "dummy" or not master_process
config_for_wandb = vars(args).copy()
wandb_run = (DummyWandb() if use_dummy_wandb
             else wandb.init(project="lora-regularization", name=args.run, config=config_for_wandb))


# -----------------------------------------------------------------------------
# Load base model

print0(f"Loading base model: tag={args.base_model_tag}, step={args.base_model_step}")
model, tokenizer, meta = load_model("base", device, "train",
                                     model_tag=args.base_model_tag, step=args.base_model_step)
token_bytes = get_token_bytes(device=device)


# -----------------------------------------------------------------------------
# Configure the base model's drop-path. For L0-L5 we disable it (the backbone should not drop
# residual branches during fine-tuning, whatever it was pretrained with). For L6 we set a linear
# schedule across depth based on --rate, exactly like pretraining, and leave the adapters noise-free.
# All of this runs on the unwrapped model (orig_model below); DDP wrapping
# happens after LoRA surgery so that DDP captures the final parameter set.
n_layer = model.config.n_layer
if args.rung == "L6":
    base_dp_rate = args.rate
    lora_rung, lora_rate = "L0", 0.0
else:
    base_dp_rate = 0.0
    lora_rung, lora_rate = args.rung, args.rate
model.drop_path_rates = [base_dp_rate * i / max(n_layer - 1, 1) for i in range(n_layer)]
print0(f"base drop_path_rates: {[round(r, 4) for r in model.drop_path_rates]}")


# -----------------------------------------------------------------------------
# Apply LoRA (still on the unwrapped model)

target_substrings = [s.strip() for s in args.lora_targets.split(",") if s.strip()]
replaced = apply_lora(model, targets=target_substrings,
                      rank=args.rank, alpha=args.alpha,
                      rung=lora_rung, rate=lora_rate, schedule=args.schedule, noise=args.noise,
                      freeze_others=True)
print0(f"Wrapped {len(replaced)} linears with LoRA  (schedule={args.schedule})")

# Keep a handle to the unwrapped model for everything that needs to access
# nanochat-specific attributes (config, drop_path_rates) or walk the LoRA
# module tree, plus for KV-cache inference (Engine). Forward passes during
# training go through `model` (compiled and possibly DDP-wrapped).
orig_model = model
# dynamic=True since SFT batches have variable length (per-batch dynamic padding).
# First training step pays a ~30s compile cost; subsequent steps are faster.
model = torch.compile(model, dynamic=True)
print0("Compiled model with torch.compile(dynamic=True)")
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    print0(f"Wrapped in DDP across world_size={ddp_world_size}")

lora_params = get_lora_params(orig_model)
n_lora_params = sum(p.numel() for p in lora_params)
print0(f"Trainable LoRA params: {n_lora_params:,}")


# -----------------------------------------------------------------------------
# Optimizer + LR schedule

optimizer = torch.optim.AdamW(lora_params, lr=args.lr,
                              betas=(0.9, 0.95), eps=1e-8,
                              weight_decay=args.weight_decay)


def lr_at(step: int) -> float:
    """Linear warmup -> cosine decay to args.final_lr_frac * args.lr."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.num_iterations - args.warmup_steps)
    progress = min(1.0, progress)
    return args.lr * (args.final_lr_frac
                       + (1 - args.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


# -----------------------------------------------------------------------------
# Data

# Effective batch (examples per optimizer step) = device_batch * world_size * grad_accum.
per_step_examples = args.device_batch_size * ddp_world_size
assert args.effective_batch_size % per_step_examples == 0, (
    f"effective batch ({args.effective_batch_size}) must be divisible by "
    f"device_batch * world_size ({per_step_examples})"
)
grad_accum_steps = args.effective_batch_size // per_step_examples
print0(f"effective_batch_size={args.effective_batch_size}, device_batch_size={args.device_batch_size}, "
       f"world_size={ddp_world_size}, grad_accum_steps={grad_accum_steps}")

BUILDERS = {"opencode": build_opencode_sft_dataset, "gsm8k": build_gsm8k_sft_dataset,
            "glaive": build_glaive_sft_dataset}
train_iter, val_iter = BUILDERS[args.dataset](
    tokenizer, args.max_seq_len, args.device_batch_size, device,
    seed=args.seed, rank=ddp_rank, world_size=ddp_world_size,
)

# Cache the first val batch right now (before any eval consumes from val_iter).
# This makes the end-of-run generation samples reproducible across all runs sharing
# the same (dataset, seed, device_batch_size): same prompts every time.
sample_batch_inputs, sample_batch_targets = next(val_iter)
print0(f"Cached {sample_batch_inputs.size(0)} val examples for end-of-run sampling.")


# -----------------------------------------------------------------------------
# Eval helpers

@torch.no_grad()
def evaluate():
    model.eval()
    # Each rank computes sums over its own data shard; we all-reduce at the end.
    sums = torch.zeros(3, device=device, dtype=torch.float64)  # [loss_sum, n_valid_tokens, n_valid_bytes]
    for _ in range(args.eval_batches):
        x, y = next(val_iter)
        loss_sum = model(x, targets=y, loss_reduction='sum')
        valid_mask = (y != PROMPT_MASK_VALUE)
        valid_token_ids = y[valid_mask]
        n_bytes = token_bytes[valid_token_ids].sum()
        sums[0] += loss_sum.detach().double()
        sums[1] += valid_mask.sum().double()
        sums[2] += n_bytes.detach().double()
    if ddp:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    model.train()
    total_loss_sum, total_valid_tokens, total_valid_bytes = sums.tolist()
    val_loss = total_loss_sum / max(1.0, total_valid_tokens)
    val_ppl = math.exp(val_loss)
    val_bpb = total_loss_sum / max(1.0, total_valid_bytes * math.log(2))
    return val_loss, val_ppl, val_bpb


def compute_lora_norms():
    """Per-layer ||A||, ||B||, ||BA||, and effective rank of BA on the unwrapped model.
    Adapter params are DDP-synchronized so any rank sees the same values; safe to compute on master.

    Effective rank is exp(H(σ_normalized)) where H is the entropy of the normalized
    singular values of BA (Roy & Vetterli 2007). For a rank-r matrix, erank ∈ [1, r];
    erank = r when singular values are uniform, erank → 1 when one direction dominates.
    """
    norms = []
    for module_name, module in orig_model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                a = module.lora_A.detach().float()
                b = module.lora_B.detach().float()
                ba = b @ a
                sv = torch.linalg.svdvals(ba)
                sv_total = sv.sum().clamp(min=1e-12)
                p = sv / sv_total
                # entropy with safe log on zeros (treat 0*log0 as 0)
                entropy = -(p * torch.where(p > 0, p.log(), torch.zeros_like(p))).sum()
                erank = entropy.exp().item()
            norms.append({
                "name": module_name,
                "A_norm": a.norm().item(),
                "B_norm": b.norm().item(),
                "BA_norm": ba.norm().item(),
                "effective_rank": erank,
            })
    return norms


@torch.no_grad()
def generate_val_samples(sample_inputs, sample_targets,
                          n_samples: int = 5, max_new_tokens: int = 256,
                          temperature: float = 0.7, seed: int = 42):
    """Sample completions on a fixed batch of held-out val prompts using the trained adapters.

    `sample_inputs` and `sample_targets` are the (B, T) tensors from a single
    cached val batch; they're captured once at init so all runs sharing the same
    (dataset, seed, device_batch_size) sample on the same prompts.

    Recovers the prompt portion from `(inputs, targets)` via the loss mask: the
    first non-(-1) position in `targets` is `prompt_length - 1`, the last
    non-(-1) is `actual_length - 2`. So `inputs[:prompt_length]` is the prompt
    and `targets[prompt_length-1 : actual_length-1]` is the ground-truth response.

    Uses Engine for fast KV-cache inference (~30 sec total for 5 samples × 256 tokens
    on the GB10). Master-only so we don't waste compute on non-rank-0 in DDP.
    """
    if not master_process:
        return []
    engine = Engine(orig_model, tokenizer)
    orig_model.eval()
    samples = []
    n_taken = min(n_samples, sample_inputs.size(0))
    for j in range(n_taken):
        targets_j = sample_targets[j]
        valid_positions = (targets_j != PROMPT_MASK_VALUE).nonzero(as_tuple=True)[0]
        if valid_positions.numel() == 0:
            continue
        first_valid = valid_positions[0].item()
        last_valid = valid_positions[-1].item()
        prompt_tokens = sample_inputs[j, : first_valid + 1].tolist()
        gt_response_tokens = targets_j[first_valid : last_valid + 1].tolist()
        generated_tokens = []
        for token_column, _masks in engine.generate(
            prompt_tokens, num_samples=1, max_tokens=max_new_tokens,
            temperature=temperature, seed=seed + j,
        ):
            generated_tokens.append(token_column[0])
        samples.append({
            "prompt": tokenizer.decode(prompt_tokens),
            "ground_truth_response": tokenizer.decode(gt_response_tokens),
            "generated_response": tokenizer.decode(generated_tokens),
        })
    orig_model.train()
    return samples


# -----------------------------------------------------------------------------
# Train

train_log_rows = []
lora_norms_rows = []  # one row per (eval step × LoRA module): trajectory of A/B/BA norms + erank
t_start = time.time()

# Eval before training begins, so val curves have a step=0 baseline (B=0 at init,
# so this measures the unmodified base model's val loss on the SFT distribution).
val_loss, val_ppl, val_bpb = evaluate()
train_log_rows.append({"step": 0, "kind": "eval", "lr": lr_at(0),
                       "val_loss": val_loss, "val_ppl": val_ppl, "val_bpb": val_bpb,
                       "wall_time": time.time() - t_start})
wandb_run.log({"val/step": 0, "val/loss": val_loss, "val/ppl": val_ppl, "val/bpb": val_bpb}, step=0)
if master_process:
    for n in compute_lora_norms():
        lora_norms_rows.append({"step": 0, **n})
print0(f"  [eval@start] val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  val_bpb={val_bpb:.4f}")

print0(f"Starting training: {args.num_iterations} iterations")

for step in range(args.num_iterations):
    step_t0 = time.time()
    # LR schedule
    lr = lr_at(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.zero_grad(set_to_none=True)

    # Two-pass macro-batch with valid-token-weighted normalization.
    # Pass 1: pull all micro-batches and count valid (non-prompt) tokens; we need
    # the global denominator before any backward call so each micro-batch's gradient
    # contribution is correctly weighted by its share of valid tokens, not by 1/N.
    micro_batches = []
    valid_count = torch.zeros((), device=device, dtype=torch.float64)
    for _ in range(grad_accum_steps):
        x, y = next(train_iter)
        valid_count += (y != PROMPT_MASK_VALUE).sum().double()
        micro_batches.append((x, y))
    if ddp:
        dist.all_reduce(valid_count, op=dist.ReduceOp.SUM)
    # `valid_count` is now the global number of valid target tokens in this macro-batch.

    # Pass 2: forward + backward with `loss_sum / valid_count`, scaled by world_size to
    # cancel DDP's gradient averaging (mean across ranks). Single-GPU: world_size=1, no-op.
    running_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    for x, y in micro_batches:
        loss_sum = model(x, targets=y, loss_reduction='sum')
        loss_for_backward = loss_sum.double() * ddp_world_size / valid_count
        loss_for_backward.backward()
        running_loss_sum += loss_sum.detach().double()
    grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_norm=args.grad_clip).item()
    optimizer.step()

    # For logging: global per-token mean loss = (sum of per-rank loss sums) / valid_count.
    if ddp:
        dist.all_reduce(running_loss_sum, op=dist.ReduceOp.SUM)
    train_loss = (running_loss_sum / valid_count).item()
    step_dt = time.time() - step_t0
    # tokens/sec is global because valid_count was already all-reduced above.
    tokens_per_sec = valid_count.item() / max(step_dt, 1e-6)
    row = {"step": step, "kind": "train", "lr": lr, "train_loss": train_loss,
           "grad_norm": grad_norm, "tokens_per_sec": tokens_per_sec,
           "wall_time": time.time() - t_start}
    train_log_rows.append(row)
    wandb_run.log({"train/step": step, "train/loss": train_loss, "train/lr": lr,
                   "train/grad_norm": grad_norm, "train/tokens_per_sec": tokens_per_sec}, step=step)

    if step % 10 == 0 or step == args.num_iterations - 1:
        print0(f"step {step:5d}/{args.num_iterations}  lr={lr:.2e}  train_loss={train_loss:.4f}  "
               f"grad_norm={grad_norm:.3f}")

    if (step % args.eval_every == 0 and step > 0) or step == args.num_iterations - 1:
        val_loss, val_ppl, val_bpb = evaluate()
        eval_row = {"step": step, "kind": "eval", "lr": lr,
                    "val_loss": val_loss, "val_ppl": val_ppl, "val_bpb": val_bpb,
                    "wall_time": time.time() - t_start}
        train_log_rows.append(eval_row)
        wandb_run.log({"val/step": step, "val/loss": val_loss, "val/ppl": val_ppl, "val/bpb": val_bpb}, step=step)
        # Snapshot the LoRA module norms + effective rank at this eval point. Only master
        # writes to keep the parquet single-writer; values are DDP-identical anyway.
        if master_process:
            for n in compute_lora_norms():
                lora_norms_rows.append({"step": step, **n})
        print0(f"  [eval] val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  val_bpb={val_bpb:.4f}")
        if master_process and args.save_every > 0 and step % args.save_every == 0:
            os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
            torch.save(get_lora_state_dict(orig_model), os.path.join(out_dir, "ckpt", f"adapters_{step:05d}.pt"))


# -----------------------------------------------------------------------------
# Save outputs

t_total = time.time() - t_start
print0(f"Training complete in {t_total/60:.1f} min")

# Generate samples on the fixed val batch we cached at init.
print0("Generating samples from cached val prompts...")
samples = generate_val_samples(sample_batch_inputs, sample_batch_targets,
                                n_samples=5, max_new_tokens=256, temperature=0.7, seed=42)

if master_process:
    # Adapter-only checkpoint (from the uncompiled model, so keys have no `_orig_mod.` prefix)
    adapters_path = os.path.join(out_dir, "adapters.pt")
    torch.save(get_lora_state_dict(orig_model), adapters_path)
    print0(f"Saved adapters: {adapters_path}")

    # Summary
    final_norms = compute_lora_norms()
    summary = {
        "config": vars(args),
        "n_lora_params": n_lora_params,
        "n_lora_modules": len(replaced),
        "wrapped_modules": replaced,
        "training_seconds": t_total,
        "final_train_loss": train_log_rows[-2]["train_loss"] if len(train_log_rows) >= 2 else None,
        "final_val_loss": train_log_rows[-1]["val_loss"] if train_log_rows[-1]["kind"] == "eval" else None,
        "final_val_ppl": train_log_rows[-1].get("val_ppl"),
        "final_val_bpb": train_log_rows[-1].get("val_bpb"),
        "final_lora_norms": final_norms,
        "samples": samples,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print0(f"Saved summary: {out_dir}/summary.json (with {len(samples)} samples)")

    # Train log parquet
    if train_log_rows:
        all_keys = sorted({k for row in train_log_rows for k in row.keys()})
        cols = {k: [row.get(k) for row in train_log_rows] for k in all_keys}
        table = pa.Table.from_pydict(cols)
        pq.write_table(table, os.path.join(out_dir, "train_log.parquet"))
        print0(f"Saved train log: {out_dir}/train_log.parquet ({len(train_log_rows)} rows)")

    # LoRA per-module norm trajectories (one row per module per eval step)
    if lora_norms_rows:
        all_keys = sorted({k for row in lora_norms_rows for k in row.keys()})
        cols = {k: [row.get(k) for row in lora_norms_rows] for k in all_keys}
        table = pa.Table.from_pydict(cols)
        pq.write_table(table, os.path.join(out_dir, "lora_norms.parquet"))
        print0(f"Saved lora norms: {out_dir}/lora_norms.parquet ({len(lora_norms_rows)} rows)")

if not isinstance(wandb_run, DummyWandb):
    wandb_run.finish()
compute_cleanup()
