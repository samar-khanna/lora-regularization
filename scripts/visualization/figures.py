"""The post's figures, drawn from results/ (run `python -m scripts.results collect` first to use your own runs).

    python -m scripts.visualization.figures           # Figures 1, 3-7, A1 and B1 -> figures/<name>.{svg,png}
    python -m scripts.visualization.toy_dynamics      # Figure 2 (toy animation)
    python -m scripts.visualization.interpretability  # Figures 8-12, E1, F1

Style: black boxed axes with inward ticks, one colour + marker per method. Teal shades for the element-wise dropout
rungs, purple shades for the drop-path rungs, grey for weight decay; within a family the rungs also differ by
marker. Error bars are +-1 standard error over seeds of the seed-paired difference.
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

from scripts.results import (BASES, DATASETS, DLP, PASS_WINDOW, RATES, RESULTS, best_of_grid, load, load_curves,
                             main_grid)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "figures"
METRIC = "pass_floor"
STYLE = {
    "L0": dict(color="black", ls="-", label="No regularization"),
    "L1": dict(color="#2A9D8F", marker="o", label=r"L1  dropout$(x)$, standard LoRA dropout"),
    "L2": dict(color="#80B8AE", marker="s", label=r"L2  dropout$(Ax)$"),
    "L3": dict(color="#2A6E64", marker="^", label=r"L3  dropout$(BAx)$"),
    "L4": dict(color="#A98BD8", marker="P", label=r"L4  drop-path$(BAx)$, token DLP"),
    "L5": dict(color="#7A5BA6", marker="D", label=r"L5  drop-path$(BAx)$, sequence DLP"),
    "L6": dict(color="#4A0284", marker="v", label=r"L6  drop-path$(Wx+BAx)$"),
    "WD": dict(color="#2E2E2E", ls="--", label="Weight decay"),
}
TICK = {"L1": "L1\ndropout\n$(x)$", "L2": "L2\ndropout\n$(Ax)$", "L3": "L3\ndropout\n$(BAx)$", "L4": "L4\ntoken\nDLP",
        "L6": "L6\ndrop-path\n$(Wx{+}BAx)$", "WD": "Weight\ndecay"}
DS_NAME = {"opencode": "OpenCodeReasoning", "gsm8k": "GSM8K", "glaive": "Glaive function calling"}
BASE_NAME = {"standard": "Standard base", "droppath": "Drop-path base"}
BASE_COLOR = {"standard": "black", "droppath": "#E69F00"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "0.88",
    "grid.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "path",
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {FIG_DIR / name}.{{svg,png}}")


def top_legend(fig, handles, labels, ncol, y=1.08):
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y),
               columnspacing=1.6, handlelength=2.2)


def seed_gaps(best, base, ds, m, ref=DLP):
    """Per-seed `m - ref` at each method's best grid value."""
    t = best[(best.base == base) & (best.dataset == ds)].pivot_table(index="seed", columns="method", values=METRIC)
    return (t[m] - t[ref]).dropna()


def mean_se(s):
    return s.mean(), s.std(ddof=1) / np.sqrt(len(s))


def two_decimals(ax):
    """Same y tick format on every data plot: two decimal places, ticks on multiples of 0.01, 0.02, 0.05, 0.1, ..."""
    if ax.get_yscale() != "linear":
        return
    lo, hi = ax.get_ylim()
    loc = MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
    ticks = loc.tick_values(lo, hi)
    if len(ticks) > 1 and np.diff(ticks).min() < 0.01 - 1e-12:
        loc = MultipleLocator(0.01)
    ax.yaxis.set_major_locator(loc)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


# ---------------------------------------------------------------------------
# Figure 1: background diagram (residual block, LoRA, dropout vs drop-path masks)

D_IO, D_IOE = "#CFE1F1", "#5B8CB8"
D_W, D_WE = "#DDE3EB", "#95A2B2"
D_AB, D_ABE = "#F6BEA5", "#C2592F"
D_ACT, D_ACTE = "#F8E6A2", "#CDAC44"
D_ELEM, D_PEREX = "#2A9D8F", "#7A5BA6"
D_ARR, D_INK, D_MUTE = "#566069", "#1F2327", "#737B84"
D_F, D_FE, D_ZOOM = "#F7F8FA", "#566069", "#EEF1F4"


class Diagram:
    """Drawing primitives shared by the diagrams (Figures 1 and 3), in the data units of an equal-aspect axes."""

    def __init__(self, ax):
        self.ax = ax

    def box(self, cx, cy, w, h, fc, ec, text, fs=15, sub=None):
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, fc=fc, ec=ec, lw=1.6, zorder=3,
                                         boxstyle="round,pad=0.02,rounding_size=0.06"))
        self.ax.text(cx, cy + (0.14 if sub else 0), text, ha="center", va="center", fontsize=fs, color=D_INK,
                     zorder=4)
        if sub:
            self.ax.text(cx, cy - 0.24, sub, ha="center", va="center", fontsize=9.5, color=D_MUTE, zorder=4)

    def frame(self, x0, y0, x1, y1):
        """The branch f, drawn the same way in every diagram."""
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, fc=D_F, ec=D_FE, lw=1.8, zorder=1,
                                         boxstyle="round,pad=0.02,rounding_size=0.1"))

    def trapez(self, cx, y0, y1, w0, w1, text):
        pts = [(cx - w0 / 2, y0), (cx + w0 / 2, y0), (cx + w1 / 2, y1), (cx - w1 / 2, y1)]
        self.ax.add_patch(Polygon(pts, closed=True, fc=D_AB, ec=D_ABE, lw=1.6, zorder=3))
        self.ax.text(cx, (y0 + y1) / 2, text, ha="center", va="center", fontsize=14, color=D_INK, zorder=4)

    def arrow(self, x1, y1, x2, y2, color=D_ARR):
        self.ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, shrinkA=0, shrinkB=0,
                                         mutation_scale=12))

    def line(self, xs, ys, color=D_ARR, **kw):
        self.ax.plot(xs, ys, color=color, lw=kw.pop("lw", 1.5), zorder=kw.pop("zorder", 2), solid_capstyle="round",
                     solid_joinstyle="round", **kw)

    def plus(self, cx, cy, r=0.18):
        self.ax.add_patch(Circle((cx, cy), r, fc="white", ec=D_ARR, lw=1.5, zorder=5))
        self.ax.text(cx, cy, "+", ha="center", va="center", fontsize=13, color=D_ARR, zorder=6)


def fig_background():
    with plt.rc_context({"mathtext.fontset": "cm", "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(12.3, 4.15))
    ax.set_xlim(0, 25.0)
    ax.set_ylim(0, 8.4)
    ax.set_aspect("equal")
    ax.axis("off")
    d = Diagram(ax)
    box, frame, trapez, arrow, line, plus = d.box, d.frame, d.trapez, d.arrow, d.line, d.plus

    def title(cx, text, eq):
        ax.text(cx, 8.05, text, ha="center", va="center", fontsize=13, fontweight="bold", color=D_INK)
        ax.text(cx, 7.2, eq, ha="center", va="center", fontsize=13.5, color=D_INK)

    # (a) residual block
    cx = 2.4
    title(cx, "(a) Residual block", r"$x' = x + f(x)$")
    box(cx, 0.6, 0.9, 0.55, D_IO, D_IOE, r"$x$")
    line([cx, cx], [0.875, 1.25])
    py = 5.5
    line([cx, 0.7, 0.7], [1.25, 1.25, py])
    arrow(0.7, py, cx - 0.18, py)
    ax.text(0.4, 3.5, "skip", rotation=90, ha="center", va="center", fontsize=10, color=D_MUTE, style="italic")
    arrow(cx, 1.25, cx, 1.95)
    fx0, fy0, fx1, fy1 = cx - 1.0, 1.95, cx + 1.0, 4.25
    frame(fx0, fy0, fx1, fy1)
    ax.text(cx, 3.4, r"$f$", ha="center", va="center", fontsize=19, color=D_INK, zorder=4)
    ax.text(cx, 2.75, "attention", ha="center", va="center", fontsize=9.5, color=D_MUTE, zorder=4)
    ax.text(cx, 2.42, "or MLP", ha="center", va="center", fontsize=9.5, color=D_MUTE, zorder=4)
    arrow(cx, 4.25, cx, py - 0.18)
    ax.text(cx + 0.15, (4.25 + py - 0.18) / 2, r"$f(x)$", ha="left", va="center", fontsize=12, color=D_INK)
    plus(cx, py)
    arrow(cx, py + 0.18, cx, 6.2)
    box(cx, 6.475, 0.9, 0.55, D_IO, D_IOE, r"$x'$")

    # (b) the same branch f, zoomed in: a frozen linear map W plus the trainable low-rank path BA
    bx0, by0, bx1, by1 = 5.3, 1.4, 11.6, 4.8
    cx = (bx0 + bx1) / 2
    xw, xa = cx - 1.4, cx + 1.4
    ax.add_patch(Polygon([(fx1, fy0), (fx1, fy1), (bx0, by1), (bx0, by0)], closed=True, fc=D_ZOOM, ec="none",
                         zorder=0))
    for ya, yb in ((fy0, by0), (fy1, by1)):
        line([fx1, bx0], [ya, yb], color=D_FE, lw=1.0, zorder=1, ls=(0, (3, 2.5)))
    frame(bx0, by0, bx1, by1)
    ax.text(bx0 + 0.3, by1 - 0.38, r"$f$", ha="center", va="center", fontsize=17, color=D_INK, zorder=4)
    title(cx, r"(b) LoRA inside $f$", r"$f(x) = h = Wx + \frac{\alpha}{r}\,BAx$")
    box(cx, 0.6, 0.9, 0.55, D_IO, D_IOE, r"$x$")
    line([cx, cx], [0.875, 1.85])
    line([xw, xa], [1.85, 1.85])
    arrow(xw, 1.85, xw, 2.25)
    box(xw, 3.0, 1.2, 1.5, D_W, D_WE, r"$W$", fs=16, sub="frozen")
    arrow(xa, 1.85, xa, 2.25)
    trapez(xa, 2.25, 2.95, 1.2, 0.45, r"$A$")
    line([xa, xa], [2.95, 3.05], color=D_ABE)
    trapez(xa, 3.05, 3.75, 0.45, 1.2, r"$B$")
    ax.text(xa + 0.36, 3.0, r"rank $r$", ha="left", va="center", fontsize=9.5, color=D_MUTE)
    ax.text(xa + 0.14, 4.0, "trainable", ha="left", va="center", fontsize=9.5, color=D_MUTE)
    line([xw, xw, cx - 0.18], [3.75, 4.3, 4.3], color=D_WE)
    line([xa, xa, cx + 0.18], [3.75, 4.3, 4.3], color=D_ABE)
    plus(cx, 4.3)
    arrow(cx, 4.48, cx, 5.5)
    box(cx, 5.775, 0.9, 0.55, D_IO, D_IOE, r"$h$")

    # (c) dropout and (d) drop-path on f(x) for a batch of two examples: each row is one token's f(x), the
    # rows grouped by example, the columns its features
    def batch(x0, dropped, color):
        pitch, size, gap, y0, n_tok = 0.55, 0.47, 0.35, 1.95, 3
        top = y0 + 2 * n_tok * pitch - (pitch - size) + gap
        for e in range(2):
            ey1 = top - e * (n_tok * pitch + gap)  # top edge of this example's rows
            for t in range(n_tok):
                for c in range(8):
                    x, y = x0 + c * pitch, ey1 - (t + 1) * pitch + (pitch - size)
                    off = dropped(e * n_tok + t, c)
                    ax.add_patch(FancyBboxPatch((x, y), size, size, boxstyle="round,pad=0,rounding_size=0.05",
                                                fc="white" if off else D_ACT, ec=color if off else D_ACTE,
                                                lw=1.3 if off else 1.0, zorder=3))
            ey0 = ey1 - n_tok * pitch + (pitch - size)
            arrow(x0 - 0.25, ey1, x0 - 0.25, ey0, color=D_MUTE)
            ax.text(x0 - 0.55, (ey0 + ey1) / 2, f"example {e + 1}", rotation=90, ha="center", va="center",
                    fontsize=9.5, color=D_MUTE)
        right = x0 + 7 * pitch + size
        arrow(x0, top + 0.35, right, top + 0.35, color=D_MUTE)
        ax.text((x0 + right) / 2, top + 0.68, r"features of $f(x)$", ha="center", va="center", fontsize=10,
                color=D_MUTE)
        return (x0 + right) / 2

    pattern = [[0, 0, 1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 1],
               [0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0]]
    for x0, name, eq, what, dropped, color in (
            (13.9, "(c) Dropout", r"$x' = x + \frac{m \odot f(x)}{1-p}$", r"zero each element of $f(x)$",
             lambda r, c: pattern[r][c], D_ELEM),
            (20.4, "(d) Drop-path", r"$x' = x + \frac{m}{1-p}\,f(x)$", r"zero each example's whole $f(x)$",
             lambda r, c: r >= 3, D_PEREX)):
        mid = batch(x0, dropped, color)
        title(mid, name, eq)
        ax.text(mid, 1.5, what, ha="center", va="center", fontsize=10.5, color=D_INK)
        ax.text(mid, 1.05, r"i.i.d. with probability $p$", ha="center", va="center", fontsize=10.5,
                color=D_INK)
    save(fig, "background")


# ---------------------------------------------------------------------------
# Figure 3: the regularization ladder, drawn on Figure 1's residual block. Element-wise dropout rungs are teal
# circles and per-sequence drop-path rungs purple diamonds, the colours of Figure 1(c, d); per-token drop-path
# (token DLP) is a light-purple diamond.

D_TOKEN = "#A98BD8"  # light purple: OKLab dE >= 15.8 from D_PEREX and >= 10.2 from D_ELEM under protan/deutan/tritan
LADDER = [  # rung, label, note, mask group
    ("L1", r"dropout$(x)$", "standard LoRA dropout", "elem"),
    ("L2", r"dropout$(Ax)$", None, "elem"),
    ("L3", r"dropout$(BAx)$", None, "elem"),
    ("L4", r"drop-path$(BAx)$", "token DLP", "token"),
    ("L5", r"drop-path$(BAx)$", "sequence DLP", "seq"),
    ("L6", r"drop-path$(Wx + BAx)$", None, "seq"),
]
D_GREY, D_GREYE = "#EEF0F2", "#BEC4CC"  # the frozen path's activation Wx


def fig_ladder():
    with plt.rc_context({"mathtext.fontset": "cm", "axes.grid": False}):
        # fixed limits and a figure of the same aspect, so text widths measured below stay valid
        xmax, ylo, yhi, inch = 13.6, 0.1, 8.95, 2 / 3  # inch: figure inches per data unit
        fig, ax = plt.subplots(figsize=(xmax * inch, (yhi - ylo) * inch))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_xlim(0, xmax)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal")
        ax.axis("off")
        d = Diagram(ax)
        xs, xw, xa = 0.7, 3.0, 6.0  # skip line, frozen W, adapter
        xc = (xw + xa) / 2
        yA0, yA1, yAx, yB0, yB1, yBAx = 2.28, 2.8, 3.19, 3.82, 4.34, 4.73  # A, Ax, B, BAx (and W, Wx beside them)
        yscale, yplus, yres, yout = 5.4, 6.72, 7.9, 8.55
        fx0, fy0, fx1, fy1 = xw - 1.0, 1.5, xa + 1.0, 7.1
        bar = 0.32  # height of an activation bar
        at = {"L1": (xa, 1.98), "L2": (xa, 3.53), "L3": (xa, 5.07), "L4": (xa, 5.76), "L5": (xa, 6.36),
              "L6": (xc, 7.38)}

        # residual block x -> x + f(x), as in Figure 1(a)
        d.box(xc, 0.6, 0.9, 0.45, D_IO, D_IOE, r"$x$")
        d.line([xc, xc], [0.83, 1.15])
        d.line([xc, xs, xs], [1.15, 1.15, yres])
        d.arrow(xs, yres, xc - 0.18, yres)
        ax.text(xs - 0.3, (1.15 + yres) / 2, "skip", rotation=90, ha="center", va="center", fontsize=10,
                color=D_MUTE, style="italic")
        d.arrow(xc, 1.15, xc, fy0)
        d.frame(fx0, fy0, fx1, fy1)
        ax.text(fx0 + 0.3, fy1 - 0.38, r"$f$", ha="center", va="center", fontsize=17, color=D_INK, zorder=4)

        # inside f: frozen W beside the adapter BA, with the activation bars Wx, Ax and BAx
        d.line([xc, xc], [fy0, 1.78])
        d.line([xw, xa], [1.78, 1.78])
        d.arrow(xw, 1.78, xw, yA0)
        d.box(xw, (yA0 + yB1) / 2, 1.2, yB1 - yA0, D_W, D_WE, r"$W$", fs=16, sub="frozen")
        d.arrow(xw, yB1, xw, yBAx - bar / 2, color=D_WE)
        d.box(xw, yBAx, 1.0, bar, D_GREY, D_GREYE, r"$Wx$", fs=13)
        d.line([xw, xw, xc - 0.18], [yBAx + bar / 2, yplus, yplus], color=D_WE)
        d.arrow(xa, 1.78, xa, yA0)
        d.trapez(xa, yA0, yA1, 1.2, 0.45, r"$A$")
        d.arrow(xa, yA1, xa, yAx - bar / 2, color=D_ABE)
        d.box(xa, yAx, 0.7, bar, D_ACT, D_ACTE, r"$Ax$", fs=13)  # narrower: r entries
        d.arrow(xa, yAx + bar / 2, xa, yB0, color=D_ABE)
        d.trapez(xa, yB0, yB1, 0.45, 1.2, r"$B$")
        d.arrow(xa, yB1, xa, yBAx - bar / 2, color=D_ABE)
        d.box(xa, yBAx, 1.0, bar, D_ACT, D_ACTE, r"$BAx$", fs=13)
        d.line([xa, xa], [yBAx + bar / 2, yscale - 0.13], color=D_ABE)
        ax.add_patch(Circle((xa, yscale), 0.13, fc="white", ec=D_ABE, lw=1.5, zorder=5))
        ax.text(xa, yscale, r"$\times$", ha="center", va="center", fontsize=11, color=D_ABE, zorder=6)
        ax.text(xa + 0.25, yscale, r"$\alpha/r$", ha="left", va="center", fontsize=12, color=D_ABE)
        d.line([xa, xa, xc + 0.18], [yscale + 0.13, yplus, yplus], color=D_ABE)
        d.plus(xc, yplus)
        d.arrow(xc, yplus + 0.18, xc, yres - 0.18)
        ax.text(xc - 0.2, at["L6"][1], r"$f(x)$", ha="right", va="center", fontsize=12, color=D_INK)
        d.plus(xc, yres)
        d.arrow(xc, yres + 0.18, xc, yout - 0.225)
        d.box(xc, yout, 0.9, 0.45, D_IO, D_IOE, r"$x'$")

        # rungs: a marker at each injection point, a leader line, and a label column on the right
        xl = fx1 + 1.1
        group = {"elem": dict(marker="o", color=D_ELEM), "token": dict(marker="D", color=D_TOKEN),
                 "seq": dict(marker="D", color=D_PEREX)}

        def marker(x, y, kind, ms=9):
            g = group[kind]
            ax.plot([x], [y], ls="none", marker=g["marker"], ms=ms * (1 if kind == "elem" else 0.85),
                    mfc=g["color"], mec="white", mew=1.2, zorder=7)

        for rung, label, note, kind in LADDER:
            x, y = at[rung]
            d.line([x, xl - 0.3], [y, y], color=group[kind]["color"], lw=1.0, zorder=2, ls=(0, (2.5, 2)))
            marker(x, y, kind)
            marker(xl, y, kind, ms=8)
            ax.text(xl + 0.32, y, rung, ha="left", va="center", fontsize=12, fontweight="bold", color=D_INK)
            t = ax.text(xl + 0.95, y, label, ha="left", va="center", fontsize=12, color=D_INK)
            if note:  # placed after the label, so measure it (the limits above are final)
                end = ax.transData.inverted().transform(t.get_window_extent(fig.canvas.get_renderer()))[1, 0]
                ax.text(end + 0.25, y, note, ha="left", va="center", fontsize=10.5, color=D_MUTE, style="italic")
        for y, kind, text in ((1.2, "elem", "element-wise dropout"), (0.78, "token", "per-token drop-path"),
                              (0.36, "seq", "per-sequence drop-path")):
            marker(xl, y, kind, ms=8)
            ax.text(xl + 0.32, y, text, ha="left", va="center", fontsize=10.5, color=D_MUTE)
    save(fig, "ladder")


# ---------------------------------------------------------------------------
# Figure 4: training and validation loss during fine-tuning (standard base)


# ---------------------------------------------------------------------------
# Figure 4: training and validation loss during fine-tuning (standard base)

def fig_overfitting(curves, best):
    """Training loss (top row) and validation loss (bottom row) on the standard base, each method at its best grid
    value for each seed. Training loss is the loss of the training pass itself, so the dropout-style methods are
    measured with their noise on. Both are averaged over PASS_WINDOW evals."""
    methods = ["L0", "WD", "L1", DLP]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.55), sharex=True)
    for col, ds in enumerate(DATASETS):
        val, tr = {}, {}
        for m in methods:
            sel = best[(best.base == "standard") & (best.dataset == ds) & (best.method == m)]
            g = curves.merge(sel[["runs_dir", "run"]], on=["runs_dir", "run"]).groupby("step")
            val[m] = g.val_loss.mean().rolling(PASS_WINDOW).mean().dropna()
            tr[m] = g.train_loss.mean().rolling(PASS_WINDOW).mean().dropna()
        ax_tr, ax_val = axes[0, col], axes[1, col]
        lo = min(s.min() for s in val.values())
        l0_min = val["L0"].min()
        top = l0_min + 1.6 * (l0_min - lo)
        for m in methods:
            st = STYLE[m]
            kw = dict(color=st["color"], ls=st.get("ls", "-"), lw=1.6, zorder=3 if m == DLP else 2)
            ax_tr.plot(tr[m].index, tr[m].values, **kw)
            ax_val.plot(val[m].index, val[m].values, label=st["label"], **kw)
        ax_val.set_ylim(lo - 0.15 * (top - lo), top)
        t_lo = min(t.min() for t in tr.values())
        t_hi = max(t[t.index >= 400].max() for t in tr.values())
        ax_tr.set_ylim(max(0.0, t_lo - 0.06 * (t_hi - t_lo)), t_hi + 0.05 * (t_hi - t_lo))  # fit the data, not 0
        ax_tr.set_title(DS_NAME[ds])
        ax_val.set_xlim(0, 4500)
        ax_val.set_xticks([0, 1500, 3000, 4500])
        ax_val.set_xlabel("Training step")
    axes[0, 0].set_ylabel("Training loss (nats)")
    axes[1, 0].set_ylabel("Validation loss (nats)")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=4, y=1.06)
    for ax in np.ravel(axes):
        two_decimals(ax)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, "overfitting")


# ---------------------------------------------------------------------------
# Figure 5: every method's validation loss relative to sequence DLP on the standard base. Bars are the mean over
# seeds, dots the individual seed-paired gaps: all dots above zero is what the one-sided Wilcoxon test summarizes.

def fig_gap_to_dlp(best, base="standard"):
    methods = ["L1", "L2", "L3", "L4", "L6", "WD"]
    x = np.arange(len(methods))
    spread = np.linspace(-0.2, 0.2, 5)  # seed k sits at the same offset in every bar
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for i, m in enumerate(methods):
            gaps = seed_gaps(best, base, ds, m)
            col = STYLE[m]["color"]
            ax.bar(x[i], gaps.mean(), 0.66, color=col, edgecolor=col, linewidth=1.0, zorder=2)
            ax.scatter(x[i] + spread[: len(gaps)], gaps.values, s=16, facecolor="white", edgecolor="black",
                       linewidth=0.8, zorder=4)
        ax.axhline(0, color=STYLE[DLP]["color"], lw=1.8, zorder=3)  # DLP itself: every bar is measured from it
        ax.set_xticks(x)
        ax.set_xticklabels([TICK[m] for m in methods], fontsize=8.0)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_title(DS_NAME[ds])
    axes[0].set_ylabel("Validation loss minus\nL5 seq. DLP's (nats)")
    axes[0].set_ylim(bottom=-0.004)  # keep DLP's zero line clear of the bottom axis
    handles = [Patch(facecolor="0.55", edgecolor="0.55", label="Mean of 5 seeds"),
               Line2D([], [], ls="none", marker="o", markersize=5, mfc="white", mec="black", label="One seed"),
               Line2D([], [], color=STYLE[DLP]["color"], lw=1.8, label="L5 sequence DLP (zero)")]
    top_legend(fig, handles, [h.get_label() for h in handles], ncol=3, y=1.08)
    for ax in np.ravel(axes):
        two_decimals(ax)
    fig.tight_layout(w_pad=1.0)
    save(fig, "gap_to_dlp")


# ---------------------------------------------------------------------------
# Figure 6: validation loss relative to no regularization at each drop rate, every rung, standard base.
# Plotted as a change in loss, so lower is better, like every other loss axis in the post.

def fig_rate_sweep(grid, best, base="standard"):
    rungs = ["L1", "L2", "L3", "L4", "L5", "L6"]
    dodge = dict(zip(rungs, np.linspace(-0.009, 0.009, len(rungs))))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sub = grid[(grid.base == base) & (grid.dataset == ds)]
        l0 = sub[sub.method == "L0"].set_index("seed")[METRIC]
        for m in rungs:
            st = STYLE[m]
            stats = [mean_se((sub[(sub.method == m) & (sub.rate == r)].set_index("seed")[METRIC] - l0).dropna())
                     for r in RATES]
            mean, se = np.array(stats).T
            ax.errorbar(np.array(RATES) + dodge[m], mean, yerr=se, color=st["color"], marker=st["marker"],
                        markersize=5, lw=1.5, elinewidth=1.0, capsize=2.5, label=st["label"],
                        zorder=3 if m == DLP else 2)
        wd = best[(best.base == base) & (best.dataset == ds) & (best.method == "WD")].set_index("seed")[METRIC]
        ax.axhline((wd - l0).mean(), color=STYLE["WD"]["color"], lw=1.2, ls="--",
                   label="Weight decay (best value)", zorder=1)
        ax.axhline(0, color="black", lw=0.8, zorder=1.6)  # no regularization; above the grid (1.5)
        ax.set_xticks(RATES)
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.3"])
        ax.set_xlim(0.025, 0.325)
        ax.set_title(DS_NAME[ds])
        ax.set_xlabel("Drop rate $p$")
    axes[0].set_ylim(top=0.004)
    axes[0].set_ylabel("Validation loss minus\nno regularization's (nats)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(STYLE[m]["label"]) for m in rungs] + [labels.index("Weight decay (best value)")]
    # 7 entries in 3 columns fill column-wise: dropout rungs | DLP rungs | L6 and weight decay
    top_legend(fig, [handles[i] for i in order], [labels[i] for i in order], ncol=3, y=1.21)
    for ax in np.ravel(axes):
        two_decimals(ax)
    fig.tight_layout(w_pad=1.2)
    save(fig, "rate_sweep")


# ---------------------------------------------------------------------------
# Figure 7: L6's and standard dropout's (L1) validation loss minus sequence DLP's at the same drop rate, on both base
# models. Colour and marker = rung; solid line and filled marker = standard base, dashed line and hollow marker =
# drop-path base. L6's rate rises linearly with depth to p; the others use p in every layer.

def fig_droppath_gap(grid):
    rungs = ["L6", "L1"]
    line = {"standard": "-", "droppath": "--"}
    dodge = {("L6", "standard"): -0.006, ("L6", "droppath"): -0.002, ("L1", "standard"): 0.002,
             ("L1", "droppath"): 0.006}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for m in rungs:
            st = STYLE[m]
            for base in BASES:
                sub = grid[(grid.base == base) & (grid.dataset == ds)]
                stats = []
                for r in RATES:
                    at = sub[sub.rate == r]
                    gap = at[at.method == m].set_index("seed")[METRIC] - at[at.method == DLP].set_index("seed")[METRIC]
                    stats.append(mean_se(gap.dropna()))
                mean, se = np.array(stats).T
                ax.errorbar(np.array(RATES) + dodge[(m, base)], mean, yerr=se, color=st["color"], ls=line[base],
                            marker=st["marker"], markersize=5, mfc=st["color"] if base == "standard" else "white",
                            mec=st["color"], lw=1.6, elinewidth=1.0, capsize=2.5, zorder=3 if m == "L6" else 2)
        ax.axhline(0, color=STYLE[DLP]["color"], lw=1.3, ls=":", zorder=1.6)
        ax.set_xticks(RATES)
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.3"])
        ax.set_xlim(0.025, 0.325)
        ax.set_title(DS_NAME[ds])
        ax.set_xlabel("Drop rate $p$")
    axes[0].set_ylabel("Validation loss minus\nL5 seq. DLP's (nats)")
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], markersize=5, lw=1.6,
                      label=STYLE[m]["label"]) for m in rungs]
    handles += [Line2D([], [], color="0.35", ls="-", marker="o", markersize=5, lw=1.6, label=BASE_NAME["standard"]),
                Line2D([], [], color="0.35", ls="--", marker="o", markersize=5, mfc="white", lw=1.6,
                       label=BASE_NAME["droppath"]),
                Line2D([], [], color=STYLE[DLP]["color"], ls=":", lw=1.3, label="L5 sequence DLP (zero)")]
    top_legend(fig, handles, [h.get_label() for h in handles], ncol=3, y=1.13)
    for ax in axes:
        two_decimals(ax)
    fig.tight_layout(w_pad=1.2)
    save(fig, "droppath_gap")


# ---------------------------------------------------------------------------
# Figure A1: pretraining curves of the two base models (results/pretraining.csv; see scripts.results collect)

def fig_pretraining():
    pre = pd.read_csv(RESULTS / "pretraining.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
    for base in BASES:
        tr = pre[(pre.base == base) & (pre.kind == "train_loss")]
        va = pre[(pre.base == base) & (pre.kind == "val_bpb")]
        smooth = tr["value"].rolling(50, min_periods=1).mean()
        axes[0].plot(tr["step"], smooth, color=BASE_COLOR[base], lw=1.3, label=BASE_NAME[base])
        axes[1].plot(va["step"], va["value"], color=BASE_COLOR[base], lw=1.3, marker="o", markersize=2.5,
                     label=BASE_NAME[base])
        print(f"  pretraining {base}: final val bpb {va['value'].iloc[-1]:.4f}")
    axes[0].set_ylim(2.7, 4.2)
    axes[0].set_ylabel("Training loss (nats)")
    axes[1].set_ylim(0.85, 1.15)
    axes[1].set_ylabel("Validation bits per byte")
    for ax in axes:
        ax.set_xlim(0, 4200)
        ax.set_xlabel("Pretraining step")
        ax.legend(loc="upper right")
    for ax in np.ravel(axes):
        two_decimals(ax)
    fig.tight_layout(w_pad=2.0)
    save(fig, "pretraining")


# ---------------------------------------------------------------------------
# Figure B1: LoRA hyperparameter sweep (OpenCodeReasoning, standard base, no regularization, 3,000 steps)

def fig_lora_sweep(runs):
    """All 96 sweep runs: rows are adapter placement, columns are batch size, one line per rank."""
    df = runs[runs.experiment == "lora-hparams"].copy()
    df["place"] = np.where(df.lora_targets == "c_q,c_v", "qv", "all")
    df = df.rename(columns={"effective_batch_size": "eb", METRIC: "val"})
    print(f"  lora sweep: {len(df)} runs")
    chosen = df[(df.place == "all") & (df["rank"] == 32) & (df.lr == 3e-3) & (df.eb == 16)].val.item()
    ranks, batches = [4, 8, 16, 32], [16, 32, 64]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(ranks)))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True, sharey=True)
    for i, (place, row) in enumerate([("qv", r"LoRA on $W_q, W_v$ only"), ("all", "LoRA on all linear maps")]):
        for j, eb in enumerate(batches):
            ax = axes[i, j]
            for r, c in zip(ranks, colors):
                s = df[(df.place == place) & (df["rank"] == r) & (df.eb == eb)].sort_values("lr")
                ax.plot(s.lr, s.val, color=c, marker="o", markersize=4, lw=1.4, label=f"rank {r}")
            if place == "all" and eb == 16:
                ax.plot([3e-3], [chosen], marker="*", markersize=13, color="white", markeredgecolor="black",
                        ls="", zorder=5, label="chosen")
            ax.set_xscale("log")
            ax.set_xticks([3e-4, 1e-3, 3e-3, 1e-2])
            ax.set_xticklabels(["3e-4", "1e-3", "3e-3", "1e-2"])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(f"Batch {eb}", fontsize=10.5)
            else:
                ax.set_xlabel("Learning rate")
            if j == 0:
                ax.set_ylabel(row + "\nBest validation loss (nats)")
    # at learning rate 1e-2 all-linear LoRA reaches only 0.80-1.61; most of those points sit above the plot
    axes[0, 0].set_ylim(0.45, 0.80)
    for ax in np.ravel(axes):
        two_decimals(ax)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=5, y=1.05)
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, "lora_sweep")


def main():
    runs = load()
    curves = load_curves()
    grid = main_grid(runs)
    best = best_of_grid(grid, METRIC)
    fig_background()
    fig_ladder()
    fig_overfitting(curves, best)
    fig_gap_to_dlp(best)
    fig_rate_sweep(grid, best)
    fig_droppath_gap(grid)
    fig_pretraining()
    fig_lora_sweep(runs)


if __name__ == "__main__":
    main()
