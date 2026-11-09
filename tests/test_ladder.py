"""CPU tests of the regularization ladder in nanochat/lora.py: mask shapes, rescaling, eval-mode behavior,
the Gaussian variant, the depth schedules, and adapter state dicts.

    python -m pytest tests/test_ladder.py -q
"""
import pytest
import torch
import torch.nn as nn

from nanochat.lora import LoRALinear, apply_lora, get_lora_state_dict, load_lora_state_dict

B, T, D_IN, D_OUT, R = 64, 6, 16, 12, 4
P = 0.3


def make(rung, rate=P, noise="bernoulli"):
    torch.manual_seed(0)
    m = LoRALinear(nn.Linear(D_IN, D_OUT, bias=False), rank=R, alpha=float(R), rung=rung, rate=rate, noise=noise)
    with torch.no_grad():
        m.lora_B.normal_()
    return m


def noisy_and_clean(m, x):
    """The adapter's contribution with the rung's noise on (train mode), and without it."""
    with torch.no_grad():
        clean = m.scale * (x @ m.lora_A.T @ m.lora_B.T)
        m.train()
        noisy = m(x) - m.base(x)
    return noisy, clean


def fitted_factor(noisy, clean, dims):
    """Least-squares multiplier of `clean` over `dims` (one value per sequence or per token)."""
    return (noisy * clean).sum(dims, keepdim=True) / (clean * clean).sum(dims, keepdim=True)


x = torch.randn(B, T, D_IN)


@pytest.mark.parametrize("rung", ["L1", "L2", "L3", "L4", "L5"])
def test_eval_mode_and_zero_rate_are_noise_free(rung):
    ref = make("L0").eval()(x)
    assert torch.allclose(make(rung).eval()(x), ref)
    assert torch.allclose(make(rung, rate=0.0).train()(x), ref)


@pytest.mark.parametrize("rung", ["L1", "L2", "L3", "L4", "L5"])
def test_noise_is_on_in_training(rung):
    m = make(rung)
    assert not torch.allclose(m.train()(x), m.eval()(x))


def test_l3_masks_each_entry():
    noisy, clean = noisy_and_clean(make("L3"), x)
    big = clean.abs() > 1e-2
    f = noisy[big] / clean[big]
    assert torch.all(torch.isclose(f, torch.zeros_like(f), atol=1e-3) | torch.isclose(f, torch.full_like(f, 1 / (1 - P)), atol=1e-3))
    assert 0.6 < (f > 0.5).float().mean().item() < 0.8  # about 1 - p of entries kept


def test_l4_masks_whole_tokens():
    noisy, clean = noisy_and_clean(make("L4"), x)
    g = fitted_factor(noisy, clean, dims=(-1,))  # one value per token
    assert torch.allclose(noisy, g * clean, atol=1e-4)
    assert torch.all(torch.isclose(g, torch.zeros_like(g), atol=1e-4) | torch.isclose(g, torch.full_like(g, 1 / (1 - P)), atol=1e-4))
    assert (g.squeeze(-1).std(dim=1) > 0.1).float().mean() > 0.5  # tokens of a sequence differ


def test_l5_masks_whole_sequences():
    noisy, clean = noisy_and_clean(make("L5"), x)
    g = fitted_factor(noisy, clean, dims=(1, 2))  # one value per sequence
    assert torch.allclose(noisy, g * clean, atol=1e-4)
    assert 0.45 < (g > 0.5).float().mean().item() < 0.95  # about 1 - p of sequences keep the adapter


@pytest.mark.parametrize("rung", ["L1", "L2", "L3", "L4", "L5"])
def test_dropout_keeps_the_expected_output(rung):
    m = make(rung)
    torch.manual_seed(1)
    xs = x[:8]
    with torch.no_grad():
        m.train()
        mean = torch.stack([m(xs) for _ in range(3000)]).mean(0)
        target = m.eval()(xs)
    assert (mean - target).norm() / (target - m.base(xs)).norm() < 0.05


def test_gaussian_dlp_factor():
    noisy, clean = noisy_and_clean(make("L5", noise="gaussian"), torch.randn(4000, 2, D_IN))
    g = fitted_factor(noisy, clean, dims=(1, 2))
    assert torch.allclose(noisy, g * clean, atol=1e-3)  # one Gaussian factor per sequence
    g = g.flatten()
    assert abs(g.mean().item() - 1.0) < 0.05
    assert abs(g.var().item() - P / (1 - P)) < 0.05


def test_invalid_configurations():
    with pytest.raises(AssertionError):
        make("L1", noise="gaussian")
    with pytest.raises(AssertionError):
        make("L6")


class TinyGPT(nn.Module):
    """Module names like nanochat's GPT: transformer.h.<i>.attn.c_q etc."""

    def __init__(self, n_layer=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList()
        for _ in range(n_layer):
            blk = nn.Module()
            blk.attn = nn.Module()
            blk.attn.c_q = nn.Linear(8, 8)
            blk.attn.c_v = nn.Linear(8, 8)
            self.transformer.h.append(blk)


@pytest.mark.parametrize("schedule,expected", [("uniform", [P] * 4),
                                               ("linear_asc", [0.0, P / 3, 2 * P / 3, P]),
                                               ("linear_desc", [P, 2 * P / 3, P / 3, 0.0])])
def test_schedules(schedule, expected):
    model = TinyGPT()
    names = apply_lora(model, targets=["c_q", "c_v"], rank=2, alpha=2.0, rung="L5", rate=P, schedule=schedule)
    assert len(names) == 8
    rates = [model.transformer.h[i].attn.c_q.rate for i in range(4)]
    assert rates == pytest.approx(expected)
    assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)


def test_state_dict_round_trip_with_compile_prefix():
    a, b = TinyGPT(), TinyGPT()
    apply_lora(a, targets=["c_q"], rank=2, alpha=2.0)
    apply_lora(b, targets=["c_q"], rank=2, alpha=2.0)
    with torch.no_grad():
        for m in a.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_()
    sd = {f"_orig_mod.{k}": v for k, v in get_lora_state_dict(a).items()}  # as saved from a compiled model
    load_lora_state_dict(b, sd, strict=True)
    for k, v in get_lora_state_dict(a).items():
        assert torch.equal(v, get_lora_state_dict(b)[k])
