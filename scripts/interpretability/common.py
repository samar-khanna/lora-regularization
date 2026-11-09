"""Shared setup for the interpretability analyses ("Why does Drop-LoRA-Path work?").

They all load the standard base model with LoRA on every linear map (rank 32, alpha 32), then swap in trained
adapters from the sweep (final adapters) or from the checkpointed reruns (`python -m scripts.sweep checkpoints`).
Outputs go to results/interpretability/ by default; scripts skip any output file that already exists, so pass --out to
write somewhere new (results/ already holds the post's outputs).
"""
from pathlib import Path

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import get_base_dir
from nanochat.lora import LoRALinear, apply_lora, load_lora_state_dict
from nanochat.sft_data import build_glaive_sft_dataset, build_gsm8k_sft_dataset, build_opencode_sft_dataset
from scripts.sweep import FIXED, run_id

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "interpretability"
BASE_TAG = "standard"
TARGETS = ["c_q", "c_k", "c_v", "c_proj", "c_fc"]
BUILD = {"opencode": build_opencode_sft_dataset, "gsm8k": build_gsm8k_sft_dataset, "glaive": build_glaive_sft_dataset}
DATASETS = ("opencode", "gsm8k", "glaive")
SEEDS = (0, 1, 2, 3, 4)
# The trained adapters the analyses compare: each method at p = 0.3, weight decay at 0.4, and no regularization.
FINAL_SET = [("L0", 0.0), ("WD", 0.4), ("L1", 0.3), ("L2", 0.3), ("L3", 0.3), ("L4", 0.3), ("L5", 0.3), ("L6", 0.3)]

dev = torch.device("cuda")


def tag(method, p):
    """Name of a method's records in results/interpretability: L5_p0.3, WD_wd0.4, L0_p0.0."""
    return f"WD_wd{p}" if method == "WD" else f"{method}_p{p}"


def run_dir(dataset, method, p, seed, base=BASE_TAG):
    """Directory of a sweep run: `method` is a rung (L0-L6, rate p) or "WD" (weight decay p)."""
    c = dict(FIXED, dataset=dataset, base=base, seed=seed)
    if method == "WD":
        c.update(rung="L0", rate=0.0, weight_decay=p)
    else:
        c.update(rung=method, rate=p if method != "L0" else 0.0)
    return Path(get_base_dir()) / "lora_runs" / run_id(c)


def ckpt_dir(dataset, rung, seed, p=0.3):
    """Directory of a checkpointed rerun (python -m scripts.sweep checkpoints)."""
    c = dict(FIXED, dataset=dataset, base=BASE_TAG, seed=seed, rung=rung, rate=p)
    return Path(get_base_dir()) / "lora_runs_ckpt" / run_id(c)


def lora_model():
    """The base model with noise-free LoRA on every linear map, in eval mode; drop-path off."""
    model, tok, _ = load_model("base", dev, "eval", model_tag=BASE_TAG)
    model.drop_path_rates = [0.0] * model.config.n_layer
    apply_lora(model, targets=TARGETS, rank=32, alpha=32.0)
    model.eval()
    mods = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    return model, tok, mods


def load_adapters(path):
    """An adapter state dict, with any torch.compile `_orig_mod.` prefix removed."""
    sd = torch.load(path, map_location="cpu")
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


__all__ = ["OUT", "BASE_TAG", "TARGETS", "BUILD", "DATASETS", "SEEDS", "FINAL_SET", "dev", "tag", "run_dir",
           "ckpt_dir", "lora_model", "load_adapters", "load_lora_state_dict", "LoRALinear"]
