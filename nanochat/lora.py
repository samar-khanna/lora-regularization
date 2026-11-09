"""LoRA with the regularization ladder from "How to Regularize your LoRA Training Better".

Each LoRALinear wraps a frozen nn.Linear W and adds a trainable rank-r update:

    h = W x + (alpha / r) * B A x

The ladder puts dropout-style noise in different places during training. Rungs use the blog's numbering:

    rung  name                   noise during training                              mask
    L0    none                   -                                                  -
    L1    dropout(x)             dropout on the adapter's input x                   one value per element
    L2    dropout(Ax)            dropout on the rank-r bottleneck Ax                one value per element
    L3    dropout(BAx)           dropout on the adapter's output BAx                one value per element
    L4    token DLP              drop BAx for a whole token                         one value per token
    L5    sequence DLP ("DLP")   drop BAx for a whole sequence                      one value per sequence
    L6    drop-path(Wx + BAx)    drop the whole attention or MLP sublayer           one value per sequence

L1 is the standard `lora_dropout` of loralib and Hugging Face PEFT: the frozen path Wx still sees the clean x.
Every rung divides kept values by (1 - p), so the expected output is unchanged, and all noise is off in eval mode.
L6 is not handled in this file: it acts on the frozen weights too, through `GPT.drop_path_rates` and
`nanochat.gpt.drop_path`, with a rate that rises linearly with depth (see scripts/lora_sft.py).

Masks are drawn independently for every LoRALinear: DLP flips one coin per sequence for each adapted linear map.

noise="gaussian" (rungs L3, L4 and L5) replaces the keep/drop factor m / (1 - p) with a factor g ~ N(1, p / (1 - p))
of the same shape: same mean and variance, so the same second-order penalty, but the adapter is never removed.
L5 with Gaussian noise is the "Gaussian DLP" of the post's Appendix G (Gaussian dropout, Srivastava et al. 2014).

Typical usage:
    from nanochat.lora import apply_lora, get_lora_params, get_lora_state_dict

    apply_lora(model, targets=["c_q", "c_k", "c_v", "c_proj", "c_fc"], rank=32, alpha=32.0, rung="L5", rate=0.3)
    optimizer = torch.optim.AdamW(get_lora_params(model), lr=3e-3)
    torch.save(get_lora_state_dict(model), "adapters.pt")
"""

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import drop_path


RUNGS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6")
ADAPTER_RUNGS = ("L1", "L2", "L3", "L4", "L5")  # rungs whose noise lives inside LoRALinear
OUTPUT_RUNGS = ("L3", "L4", "L5")  # rungs that mask BAx; only these support noise="gaussian"
RUNG_NAMES = {
    "L0": "no regularization",
    "L1": "dropout(x), standard LoRA dropout",
    "L2": "dropout(Ax)",
    "L3": "dropout(BAx)",
    "L4": "drop-path(BAx) per token, token DLP",
    "L5": "drop-path(BAx) per sequence, sequence DLP",
    "L6": "drop-path(Wx + BAx), full drop-path",
}
NOISES = ("bernoulli", "gaussian")


class LoRALinear(nn.Module):
    """Frozen base nn.Linear + trainable rank-r update, with one rung's noise during training."""

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 32.0,
                 rung: str = "L0", rate: float = 0.0, noise: str = "bernoulli"):
        super().__init__()
        assert rank > 0, f"rank must be positive, got {rank}"
        assert rung == "L0" or rung in ADAPTER_RUNGS, \
            f"LoRALinear handles rungs L0-L5, got {rung!r} (L6 is applied by the model's blocks)"
        assert noise in NOISES, f"unknown noise {noise!r}"
        assert noise == "bernoulli" or rung in OUTPUT_RUNGS, \
            f"noise='gaussian' is defined for rungs {OUTPUT_RUNGS}, got {rung!r}"
        assert 0.0 <= rate < 1.0, f"rate must be in [0, 1), got {rate}"

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.rung = rung
        self.rate = float(rate)
        self.noise = noise

        # Standard PEFT init: A ~ uniform(-1/sqrt(d_in), +1/sqrt(d_in)), B = 0.
        # B = 0 means the adapter contributes 0 at init, so the model's initial
        # output equals the base model's exactly.
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.lora_A, -bound, bound)

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}, rung={self.rung}, rate={self.rate}, noise={self.noise}"

    def forward(self, x):
        y = self.base(x)
        noisy = self.training and self.rate > 0
        # L1: standard LoRA dropout, on the adapter's input. The frozen path keeps the clean x (y above).
        x_lora = x
        if noisy and self.rung == "L1":
            x_lora = F.dropout(x, p=self.rate, training=True)
        # Cast LoRA weights to match input dtype, mirroring nanochat.gpt.Linear behavior.
        Ax = F.linear(x_lora, self.lora_A.to(x.dtype))
        if noisy and self.rung == "L2":
            Ax = F.dropout(Ax, p=self.rate, training=True)
        BAx = F.linear(Ax, self.lora_B.to(x.dtype))
        if noisy and self.rung in OUTPUT_RUNGS:
            BAx = self._output_noise(BAx)
        return y + self.scale * BAx

    def _output_noise(self, BAx):
        p = self.rate
        if self.noise == "gaussian":
            # Same mask shapes as the Bernoulli rungs below, with g ~ N(1, p / (1 - p)) instead of m / (1 - p).
            if self.rung == "L3":
                shape = BAx.shape
            elif self.rung == "L4":
                shape = BAx.shape[:-1] + (1,)
            else:
                shape = (BAx.shape[0],) + (1,) * (BAx.ndim - 1)
            g = 1.0 + (p / (1.0 - p)) ** 0.5 * torch.randn(shape, device=BAx.device, dtype=torch.float32)
            return BAx * g.to(BAx.dtype)
        if self.rung == "L3":  # one mask value per element of BAx
            return F.dropout(BAx, p=p, training=True)
        if self.rung == "L4":  # one mask value per token, shared across the token's features
            keep = BAx.new_empty(BAx.shape[:-1] + (1,)).bernoulli_(1.0 - p)
            return BAx * keep / (1.0 - p)
        return drop_path(BAx, p, training=True)  # L5: one mask value per sequence, shared across tokens and features


_LAYER_RE = re.compile(r"transformer\.h\.(\d+)\.")


def _scheduled_rate(layer_idx: int, num_layers: int, peak_rate: float, schedule: str) -> float:
    """Per-layer rate for the chosen schedule. peak_rate is the rate at the extreme layer; the other
    extreme is 0. Matches the linear shape of the model's drop-path schedule (used by L6)."""
    if schedule == "uniform":
        return peak_rate
    denom = max(num_layers - 1, 1)
    if schedule == "linear_asc":
        return peak_rate * (layer_idx / denom)
    if schedule == "linear_desc":
        return peak_rate * ((num_layers - 1 - layer_idx) / denom)
    raise ValueError(f"unknown schedule: {schedule!r}")


def apply_lora(model: nn.Module, targets, rank: int = 8, alpha: float = 32.0,
               rung: str = "L0", rate: float = 0.0, schedule: str = "uniform",
               noise: str = "bernoulli", freeze_others: bool = True):
    """Replace every nn.Linear in `model` whose qualified name contains any of
    the substrings in `targets` with a LoRALinear wrapping it.

    Args:
        targets: substring or list of substrings to match against each module's
            full qualified name (e.g. ["c_q", "c_v"] for QV-only,
            ["c_q", "c_k", "c_v", "c_proj", "c_fc"] for all linear maps).
        rung, rate, noise: the adapter-level rung (L0-L5), its rate p, and its noise type.
        schedule: how to distribute the rate across layers (Appendix D).
            "uniform" (default, used everywhere in the post): every adapter gets `rate`.
            "linear_asc": layer 0 -> 0, last layer -> rate (ViT/timm convention).
            "linear_desc": layer 0 -> rate, last layer -> 0.
        freeze_others: if True, freeze every parameter in the model first, so
            that after wrapping only the LoRA A/B parameters are trainable.

    Returns the list of qualified names that were wrapped (in tree order).
    """
    if isinstance(targets, str):
        targets = [targets]

    if freeze_others:
        for p in model.parameters():
            p.requires_grad_(False)

    # Discover the layer count from module names (transformer.h.<idx>.*).
    layer_ids = set()
    for name, _ in model.named_modules():
        m = _LAYER_RE.search(name)
        if m:
            layer_ids.add(int(m.group(1)))
    num_layers = (max(layer_ids) + 1) if layer_ids else 1

    # Collect targets first; do not mutate the tree during iteration.
    to_replace = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and any(s in full_name for s in targets):
                to_replace.append((parent, child_name, child, full_name))

    replaced = []
    for parent, child_name, child, full_name in to_replace:
        m = _LAYER_RE.search(full_name)
        layer_idx = int(m.group(1)) if m else 0
        this_rate = _scheduled_rate(layer_idx, num_layers, rate, schedule)
        wrapped = LoRALinear(child, rank=rank, alpha=alpha, rung=rung, rate=this_rate, noise=noise)
        wrapped = wrapped.to(child.weight.device)
        setattr(parent, child_name, wrapped)
        replaced.append(full_name)

    return replaced


def get_lora_params(model: nn.Module):
    """Return the list of LoRA A and B parameters in tree order (for the optimizer)."""
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.append(m.lora_A)
            params.append(m.lora_B)
    return params


def get_lora_state_dict(model: nn.Module):
    """Adapter-only state dict: just the lora_A and lora_B tensors, keyed by full path.
    Pass the uncompiled model; keys from a torch.compile wrapper carry an `_orig_mod.` prefix."""
    state = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{module_name}.lora_A"] = module.lora_A.detach().cpu().clone()
            state[f"{module_name}.lora_B"] = module.lora_B.detach().cpu().clone()
    return state


def load_lora_state_dict(model: nn.Module, state_dict, strict: bool = True):
    """Load an adapter-only state dict into a model that already has LoRA applied.
    Keys saved from a torch.compile wrapper (prefix `_orig_mod.`) are accepted.

    If strict, asserts that the keys in state_dict and the model's LoRA params match exactly.
    """
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    own = {name: p for name, p in model.named_parameters()
           if ".lora_A" in name or ".lora_B" in name}
    if strict:
        missing_in_state = set(own.keys()) - set(state_dict.keys())
        unexpected_in_state = set(state_dict.keys()) - set(own.keys())
        assert not missing_in_state and not unexpected_in_state, \
            f"LoRA state dict mismatch. missing: {missing_in_state}, unexpected: {unexpected_in_state}"
    for name, tensor in state_dict.items():
        if name not in own:
            continue
        with torch.no_grad():
            own[name].copy_(tensor.to(own[name].device))
