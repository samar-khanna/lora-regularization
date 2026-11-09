"""Figures of the post's "Why does Drop-LoRA-Path work?" section and Appendices E-F, from results/interpretability/.
File names are those of the post's images.

    Figure 8   mech_forgetting     KL(base || fine-tuned) on pretraining text, every method (battery final)
    Figure 9   mech_learn_forget   KL against next-token accuracy over training (battery ckpt)
    Figure 10  mech_coadapt        validation loss with only one block's adapters kept (battery final)
    Figure 11  mech_mask_pairs     which pairs of entries share a mask value under L3, L4 and L5 (diagram)
    Figure 12  mech_pair_vs_val    each rung's penalties against its validation loss minus DLP's (battery final,
                                   results/runs.csv)
    Figure E1  mech_combine        hybrids of two runs' adapters (hybrids)
    Figure F1  mech_pair_dynamics  pair terms over training (pair_terms)

usage: python -m scripts.visualization.interpretability   -> figures/<name>.{svg,png}
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.interpretability.common import tag
from scripts.interpretability.records import (METHODS, RATE, ckpt_curve, final_records, hybrid_records, mean_se,
                                              pair_term_curve, pair_vs_val_data)
from scripts.visualization.figures import DS_NAME, STYLE, save, top_legend, two_decimals

TICK = {"L0": "No\nreg.", "WD": "Weight\ndecay", "L1": "L1\ndropout\n$(x)$", "L2": "L2\ndropout\n$(Ax)$",
        "L3": "L3\ndropout\n$(BAx)$", "L4": "L4\ntoken\nDLP", "L5": "L5\nseq.\nDLP",
        "L6": "L6\ndrop-path\n$(Wx{+}BAx)$"}
TICK_SHORT = {"L0": "No\nreg.", "WD": "WD", "L1": "L1", "L2": "L2", "L3": "L3", "L4": "L4", "L5": "L5", "L6": "L6"}
LEGEND_NOISE = {"L3": r"L3 dropout$(BAx)$", "L4": "L4 token DLP", "L5": "L5 sequence DLP"}
LEGEND_PAIR = {"L1": r"L1 dropout$(x)$", "L2": r"L2 dropout$(Ax)$", "L3": r"L3 dropout$(BAx)$", "L4": "L4 token DLP",
               "L5": "L5 sequence DLP"}


def style(r):
    return STYLE[r]


# ---------------------------------------------------------------------------
# Figure 8: KL from the base model on held-out pretraining text, per method (bars = mean, dots = seeds).

def fig_forgetting(datasets=("gsm8k",)):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.4, 3.4) if n == 1 else (10.5, 2.85), squeeze=False)  # n = 3: 4:1
    for ax, ds in zip(axes[0], datasets):
        rec = final_records(ds)
        x = np.arange(len(METHODS))
        for i, r in enumerate(METHODS):
            v = np.array([q["pre_kl"] for q in rec[r]])
            col = style(r)["color"]
            ax.bar(x[i], v.mean(), 0.66, color=col if r != "L0" else "0.15", edgecolor=col, zorder=2)
            ax.scatter(x[i] + np.linspace(-0.2, 0.2, len(v)), v, s=16, facecolor="white", edgecolor="black",
                       linewidth=0.8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([TICK[r] if n == 1 else TICK_SHORT[r] for r in METHODS], fontsize=7.5 if n == 1 else 8.5)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.set_ylim(0, None)
        ax.set_title(DS_NAME[ds])
    axes[0, 0].set_ylabel(r"$\mathrm{KL}(p_\theta \,\Vert\, p_{\theta'})$ (nats per token)")
    handles = [Patch(facecolor="0.55", edgecolor="0.55", label="Mean over seeds"),
               Line2D([], [], ls="none", marker="o", markersize=5, mfc="white", mec="black", label="One seed")]
    top_legend(fig, handles, [h.get_label() for h in handles], ncol=2, y=1.08)
    for ax in axes[0]:
        two_decimals(ax)
    fig.tight_layout()
    save(fig, "mech_forgetting")

# ---------------------------------------------------------------------------
# Figure 9: KL from the base model on pretraining text against next-token accuracy on the validation set,
# every 250 steps of the checkpointed reruns. Mean over seeds.


def learn_forget_datasets(rungs=("L1", "L5"), n_seeds=5):
    """Datasets whose checkpoint data are complete for every rung."""
    out = []
    for ds in ("opencode", "gsm8k", "glaive"):
        curves = [ckpt_curve(r, ds) for r in rungs]
        if all(c is not None and c[2] >= n_seeds for c in curves):
            out.append(ds)
    return tuple(out)

def fig_learn_forget(datasets=("gsm8k",), rungs=("L1", "L5")):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.4, 3.7) if n == 1 else (10.5, 2.85), squeeze=False)
    n_seeds = []
    for ax, ds in zip(axes[0], datasets):
        for r in rungs:
            c = ckpt_curve(r, ds)
            if c is None:
                continue
            steps, m, k = c
            n_seeds.append((ds, r, k))
            st = style(r)
            ax.plot(100 * m[:, 0], m[:, 1], color=st["color"], marker=st.get("marker", "o"), markersize=3.5,
                    lw=1.6, label=st["label"])
            j = int(np.argmin(m[:, 2]))
            ax.scatter([100 * m[j, 0]], [m[j, 1]], s=110, facecolor="none", edgecolor=st["color"], linewidth=1.5,
                       zorder=5)
        ax.set_xlabel("Next-token accuracy on validation (%)" if n > 1 else
                      f"Next-token accuracy on {DS_NAME[ds]} validation (%)")
        ax.set_title(DS_NAME[ds])
        two_decimals(ax)
    axes[0, 0].set_ylabel(r"$\mathrm{KL}(p_\theta \,\Vert\, p_{\theta'})$ (nats per token)")
    if n == 1:
        axes[0, 0].legend(loc="upper left", fontsize=8)
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        top_legend(fig, handles, labels, ncol=2, y=1.08)
    fig.tight_layout()
    save(fig, "mech_learn_forget")
    return n_seeds

# ---------------------------------------------------------------------------
# Figure 10: validation loss with only one block's adapters kept (all others removed), minus the base model's.

def fig_coadapt(datasets=("gsm8k",), rungs=("L0", "L1", "L5")):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.2, 3.4) if n == 1 else (10.5, 2.85), squeeze=False)
    blocks = np.arange(12)
    for ax, ds in zip(axes[0], datasets):
        rec = final_records(ds)
        for r in rungs:
            st = style(r)
            alone = [np.array(q["only_val"]) - q["base_blk_val"] for q in rec[r]]
            m, se = mean_se(alone)
            ax.errorbar(blocks, m, yerr=se, color=st["color"], marker=st.get("marker", "o"), markersize=4.5, lw=1.6,
                        elinewidth=1.0, capsize=2.5, label=st["label"])
        ax.axhline(0, color="black", lw=0.9, ls="--", zorder=1.5, label="Base model")
        ax.set_xticks(blocks if n == 1 else blocks[::2])
        ax.set_xlabel("Block whose adapters are kept")
        ax.set_title(DS_NAME[ds])
        two_decimals(ax)
    axes[0, 0].set_ylabel("Validation loss minus\nthe base model's (nats)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(l) for l in labels if l != "Base model"] + [labels.index("Base model")]
    top_legend(fig, [handles[i] for i in order], [labels[i] for i in order], ncol=2 if n == 1 else 4,
               y=1.13 if n == 1 else 1.08)
    fig.tight_layout()
    save(fig, "mech_coadapt")

# ---------------------------------------------------------------------------
# Figure E1: validation loss of a hybrid built from two runs of the same method (blocks 0-5 from one seed,
# blocks 6-11 from another) minus the mean of the two originals'. No regularization is off the scale.

def fig_combine(datasets=("opencode", "gsm8k", "glaive"), rungs=("WD", "L1", "L2", "L3", "L4", "L5", "L6")):
    """One row per dataset, stacked, since Figure E1 sits in the appendix's collapsible block (text-column width).
    OpenCodeReasoning's pairs are scored on the problems both runs held out (hybrids.py)."""
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(5.6, 1.95 * n + 0.9), sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(rungs))
    spread = np.linspace(-0.2, 0.2, 5)
    off_scale = {}
    for ax, ds in zip(axes[:, 0], datasets):
        m = hybrid_records(ds)
        for i, r in enumerate(rungs):
            gaps = np.array([q["hybrid"] - q["ref"] for q in m[tag(r, RATE[r])]])
            col = style(r)["color"]
            ax.bar(x[i], gaps.mean(), 0.66, color=col, edgecolor=col, zorder=2)
            ax.scatter(x[i] + spread[: len(gaps)], gaps, s=16, facecolor="white", edgecolor="black", linewidth=0.8,
                       zorder=4)
        off_scale[ds] = float(np.mean([q["hybrid"] - q["ref"] for q in m[tag("L0", RATE["L0"])]]))
        ax.axhline(0, color="black", lw=0.8, zorder=3)
        ax.set_xlim(-0.6, len(rungs) - 0.4)
        ax.set_title(DS_NAME[ds], fontsize=10)
        ax.tick_params(axis="x", length=0)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels([TICK[r] for r in rungs], fontsize=8)
    axes[n // 2, 0].set_ylabel("Validation loss, hybrid\nminus the originals' (nats)")
    handles = [Patch(facecolor="0.55", edgecolor="0.55", label="Mean of 5 seed pairs"),
               Line2D([], [], ls="none", marker="o", markersize=5, mfc="white", mec="black", label="One seed pair")]
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.6, handlelength=2.2, frameon=False)
    for ax in axes[:, 0]:
        two_decimals(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.8)
    save(fig, "mech_combine")
    return off_scale

# ---------------------------------------------------------------------------
# Figure 11: which pairs of entries share a mask value (E_m[eps_i eps_j] = sigma^2) under L3, L4 and L5.

def fig_mask_pairs(n_tok=3, n_feat=3):
    """Small, with small tick and axis labels; thick lines separate tokens."""
    n = n_tok * n_feat
    tok = np.repeat(np.arange(n_tok), n_feat)
    pats = [("(a) L3: per entry", np.eye(n, dtype=bool), style("L3")["color"]),
            ("(b) L4: per token", tok[:, None] == tok[None, :], style("L4")["color"]),
            ("(c) L5: per sequence", np.ones((n, n), dtype=bool), style("L5")["color"])]
    fig, axes = plt.subplots(1, 3, figsize=(4.3, 1.75))
    centers = np.arange(n_tok) * n_feat + (n_feat - 1) / 2
    for k, (ax, (title, pat, col)) in enumerate(zip(axes, pats)):
        ax.imshow(pat, cmap=mpl.colors.ListedColormap(["white", col]), vmin=0, vmax=1, interpolation="nearest")
        ax.grid(False)
        for e in np.arange(-0.5, n, 1):
            ax.axhline(e, color="0.85", lw=0.4)
            ax.axvline(e, color="0.85", lw=0.4)
        for e in np.arange(n_feat - 0.5, n - 1, n_feat):
            ax.axhline(e, color="0.3", lw=1.0)
            ax.axvline(e, color="0.3", lw=1.0)
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels([f"token {t + 1}" for t in range(n_tok)], fontsize=5)
        ax.set_yticklabels([f"token {t + 1}" for t in range(n_tok)], fontsize=5, rotation=90, va="center")
        ax.tick_params(length=0, pad=1.5)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("entry $j$", fontsize=6.5, labelpad=1.5)
        if k == 0:
            ax.set_ylabel("entry $i$", fontsize=6.5, labelpad=1.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, "mech_mask_pairs")

# ---------------------------------------------------------------------------
# Figure 12: each rung's validation loss minus DLP's (sweep runs at p = 0.3, paired by seed) against its
# penalties, measured on the same runs' final adapters as the rise in training loss under each noise.

def fig_pair_vs_val(datasets=("opencode", "gsm8k", "glaive")):
    d = pair_vs_val_data(datasets)
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 2.9), sharey=True)
    gaps = np.array([v["gap"][0] for v in d.values()])
    for ax, key, title, xlabel in (
            (axes[0], "per", "(a) Per-entry penalty", r"$\Omega_{\mathrm{L3}}$ of the trained adapters (nats)"),
            (axes[1], "tok", "(b) Token DLP's penalty", r"$\Omega_{\mathrm{L4}}$ of the trained adapters (nats)"),
            (axes[2], "full", "(c) Sequence DLP's penalty", r"$\Omega_{\mathrm{L5}}$ of the trained adapters (nats)"),
            (axes[3], "pair", "(d) Non-diagonal pair terms only",
             r"$\Omega_{\mathrm{L5}} - \Omega_{\mathrm{L3}}$ of the trained adapters (nats)")):
        for (ds, r), v in d.items():
            st = style(r)
            (xm, xs), (ym, ys) = v[key], v["gap"]
            ax.errorbar(xm, ym, xerr=xs, yerr=ys, color=st["color"], marker=st.get("marker", "o"), markersize=6.5,
                        ls="none", elinewidth=0.9, capsize=2, zorder=3,
                        label=LEGEND_PAIR[r] if ds == datasets[0] else None)
        xs = np.array([v[key][0] for v in d.values()])
        slope, icpt = np.polyfit(xs, gaps, 1)
        xx = np.array([xs.min(), xs.max()])
        ax.plot(xx, icpt + slope * xx, color="0.6", lw=1.0, ls="--", zorder=1)
        ax.text(0.96, 0.06, f"Pearson $r$ = {np.corrcoef(xs, gaps)[0, 1]:.2f}", transform=ax.transAxes, fontsize=9,
                color="0.25", ha="right", va="bottom")
        ax.axhline(0, color="0.8", lw=0.8, zorder=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.grid(color="0.92", lw=0.6, zorder=0)
    axes[0].set_ylabel("Validation loss minus\nDLP's (nats)")
    axes[0].set_yticks([0, 0.005, 0.01, 0.015])
    axes[0].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))  # gaps are below 0.02 nats
    axes[0].set_xticks([0.05, 0.10, 0.15, 0.20])
    axes[0].set_xlim(0.025, 0.215)
    for ax in axes[1:3]:
        ax.set_xticks([0.05, 0.15, 0.25])
        ax.set_xlim(0.025, 0.33)
    axes[3].set_xticks([0, 0.05, 0.10, 0.15])
    axes[3].set_xlim(-0.025, 0.165)
    for ax in axes:
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.14), columnspacing=2.0,
               handlelength=1.8, title="Adapters trained with", title_fontsize=9.5, frameon=False)
    fig.tight_layout(w_pad=2.0)
    save(fig, "mech_pair_vs_val")
    return d

# ---------------------------------------------------------------------------
# Figure F1: pair terms (Omega_L5 - Omega_L3 on training batches) at each checkpoint of the standard-dropout
# and DLP reruns. Dotted lines: each method's lowest validation loss (Figure 9 data).


def fig_pair_dynamics(datasets=("opencode", "gsm8k", "glaive"), rungs=("L1", "L5")):
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.6, 2.45), sharey=True)
    for ax, ds in zip(axes, datasets):
        for r in rungs:
            st = style(r)
            steps, v = pair_term_curve(r, ds)
            m, se = mean_se(v)
            ax.errorbar(steps, m, yerr=se, color=st["color"], marker=st.get("marker", "o"), markersize=3.5, lw=1.4,
                        elinewidth=0.8, capsize=2, label=LEGEND_NOISE.get(r, st["label"]) if r != "L1" else
                        r"L1 dropout$(x)$, standard", zorder=3)
            c = ckpt_curve(r, ds)
            if c is not None:
                best_step = c[0][int(np.argmin(c[1][:, 2]))]
                ax.axvline(best_step, color=st["color"], lw=1.0, ls=":", zorder=1)
        ax.axhline(0, color="0.6", lw=0.8, zorder=0.5)
        ax.set_title(DS_NAME[ds], fontsize=9)
        ax.set_xlabel("Training step", fontsize=8.5)
        ax.set_xticks([0, 1500, 3000, 4500])
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(r"$\Omega_{\mathrm{L5}} - \Omega_{\mathrm{L3}}$ (nats)", fontsize=8.5)
    two_decimals(axes[0])
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", lw=1.0, ls=":"))
    labels.append("Lowest validation loss")
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1), fontsize=8, frameon=False,
               columnspacing=1.4, handlelength=2.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, "mech_pair_dynamics")


def main():
    datasets = ("opencode", "gsm8k", "glaive")
    fig_forgetting(datasets)
    fig_learn_forget(learn_forget_datasets() or ("gsm8k",))
    fig_coadapt(datasets)
    fig_mask_pairs()
    fig_pair_vs_val(datasets)
    fig_combine(datasets)
    fig_pair_dynamics(datasets)


if __name__ == "__main__":
    main()
