"""Figure 2, the toy illustration in the post's Background: how no regularization, weight decay, dropout and drop-path
change what a small residual MLP learns from a few noisy points.

All four models start from the same initialization and see the same 30 points of sin(2*pi*u) plus noise.
They are trained in lockstep (one batched model per config), full batch, Adam with a cosine schedule.
Dropout and drop-path act on each residual branch's output exactly as in the post's equations:
x' = x + m * f(x) / (1 - p), with m per element (dropout) or per example (drop-path).

Writes figures/toy_dynamics.{svg,png} (static: final fits + loss curves) and, on request, the same panels
animated over training: toy_dynamics.gif, and a looping video (toy_dynamics.mp4 in H.264, toy_dynamics.webm in
VP9, toy_dynamics_poster.png = the last frame).
The video needs ffmpeg on PATH. Each training run is cached in figures/<name>_rec.npz; --cached redraws
from it (about 8 minutes faster on CPU) and refuses if the training settings differ.

Usage:
    python -m scripts.visualization.toy_dynamics [--gif] [--video] [--cached]
"""

import argparse
import json
import math
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from PIL import Image

from scripts.visualization.figures import FIG_DIR, save

KINDS = {"none": 0, "wd": 0, "dropout": 1, "droppath": 2}
FAMILIES = ("units", "blocks")
# Loss curves of the two sub-network families: line style plus a marker, offset so both show where they overlap
FAM_STYLE = {"units": dict(ls=(0, (5, 2.5)), marker="o", markevery=(5, 10), mfc="white"),
             "blocks": dict(ls=(0, (0.1, 2.2)), dash_capstyle="round", marker="^", markevery=(0, 10))}
COLOR = {"none": "black", "wd": "0.4", "dropout": "#2A9D8F", "droppath": "#7A5BA6"}  # Figure 1's teal/purple
TITLE = {"none": "No regularization", "wd": "Weight decay", "dropout": "Dropout", "droppath": "Drop-path"}


def target(u):
    return torch.sin(2 * math.pi * u)


class Stack:
    """C independent residual MLPs trained in lockstep, one per regularizer config (same init, same data)."""

    def __init__(self, cfgs, d, hid, depth, lr, ln, seed, dev):
        self.C, self.d, self.depth, self.lr, self.ln, self.dev = len(cfgs), d, depth, lr, ln, dev
        self.kind = torch.tensor([KINDS[c["kind"]] for c in cfgs], device=dev)
        self.pr = torch.tensor([c.get("p", 0.0) for c in cfgs], device=dev)
        self.wd = torch.tensor([c.get("wd", 0.0) for c in cfgs], device=dev)
        g = torch.Generator().manual_seed(seed)

        def lin(o, i):  # nn.Linear's default init
            b = 1 / math.sqrt(i)
            return (torch.rand(o, i, generator=g) * 2 - 1) * b, (torch.rand(o, generator=g) * 2 - 1) * b

        iw, ib = lin(d, 1)
        blocks = [lin(hid, d) + lin(d, hid) for _ in range(depth)]
        ow, ob = lin(1, d)

        def rep(t):
            return t.unsqueeze(0).repeat(self.C, *[1] * t.ndim).to(dev).requires_grad_()

        self.P = dict(iw=rep(iw), ib=rep(ib),
                      w1=rep(torch.stack([b[0] for b in blocks])), b1=rep(torch.stack([b[1] for b in blocks])),
                      w2=rep(torch.stack([b[2] * 0.5 for b in blocks])), b2=rep(torch.zeros(depth, d)),
                      ow=rep(ow), ob=rep(ob))
        self.opt = torch.optim.Adam(self.P.values(), lr=lr)

    def forward(self, u, masks=None):
        P = self.P
        h = torch.einsum("cdi,ni->cnd", P["iw"], u) + P["ib"][:, None]
        for l in range(self.depth):
            z = F.layer_norm(h, (self.d,)) if self.ln else h
            z = F.relu(torch.einsum("chd,cnd->cnh", P["w1"][:, l], z) + P["b1"][:, l, None])
            fx = torch.einsum("cdh,cnh->cnd", P["w2"][:, l], z) + P["b2"][:, l, None]
            h = h + (fx if masks is None else fx * masks[l])
        return torch.einsum("cod,cnd->cno", P["ow"], h)[..., 0] + P["ob"]

    def masks(self, n, shared=False):
        """Masks scaled by 1/(1-p): per element for dropout, per example for drop-path, ones otherwise.
        shared=True draws one sub-network per config (the same mask for every input), for plotting."""
        keep = (1 - self.pr)[None, :, None, None]
        rows = 1 if shared else n
        el = (torch.rand(self.depth, self.C, rows, self.d, device=self.dev) < keep).float()
        ex = (torch.rand(self.depth, self.C, rows, 1, device=self.dev) < keep).float().expand(-1, -1, -1, self.d)
        k = self.kind[None, :, None, None]
        return torch.where(k == 1, el, torch.where(k == 2, ex, torch.ones_like(el))) / keep

    def view_masks(self, kind, p, gen):
        """One random sub-network, the same for every config: random units (kind="units") or random blocks
        (kind="blocks") removed with probability p, survivors scaled by 1/(1-p)."""
        cols = self.d if kind == "units" else 1
        m = (torch.rand(self.depth, 1, 1, cols, generator=gen) >= p).float().expand(-1, -1, -1, self.d)
        return (m / (1 - p)).to(self.dev).expand(-1, self.C, -1, -1)

    def step(self, u, y, lr):
        loss = ((self.forward(u, self.masks(len(u))) - y) ** 2).mean(1).sum()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():  # decoupled weight decay on the weight matrices, as in AdamW
            for k in ("iw", "w1", "w2", "ow"):
                t = self.P[k]
                t.mul_(1 - lr * self.wd.view(-1, *[1] * (t.ndim - 1)))
        for grp in self.opt.param_groups:
            grp["lr"] = lr
        self.opt.step()

    @torch.no_grad()
    def predict(self, u, masks=None):
        return self.forward(u, masks)


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfgs = [dict(kind="none"), dict(kind="wd", wd=args.wd),
            dict(kind="dropout", p=args.p_dropout), dict(kind="droppath", p=args.p_droppath)]
    g = torch.Generator().manual_seed(args.data_seed)
    u = torch.rand(args.n, 1, generator=g)
    y = target(u)[:, 0] + args.sigma * torch.randn(args.n, generator=g)
    uv = torch.rand(4000, 1, generator=g)
    yv = target(uv)[:, 0] + args.sigma * torch.randn(4000, generator=g)
    ug = torch.linspace(0, 1, 300)[:, None]
    u, y, uv, yv, ug = (t.to(dev) for t in (u, y, uv, yv, ug))

    st = Stack(cfgs, args.d, args.hid, args.depth, args.lr, args.ln, args.seed, dev)
    # Fixed sub-networks, the same in every column: random units (what dropout drops) or random whole blocks
    # (what drop-path drops) removed at rate p. The first k_sub are drawn; all k_eval enter the loss average.
    mg = torch.Generator().manual_seed(args.seed + 99)
    torch.manual_seed(args.seed + 1)  # the training masks
    views = {fam: [st.view_masks(fam, args.p_view, mg) for _ in range(args.k_eval)] for fam in FAMILIES}
    rec = {k: [] for k in ("step", "train", "val", "fit") + tuple(f"sub_{f}" for f in FAMILIES)
           + tuple(f"val_{f}" for f in FAMILIES)}
    for t in range(args.steps + 1):
        if t % args.every == 0:
            rec["step"].append(t)
            rec["train"].append(((st.predict(u) - y) ** 2).mean(1).cpu().numpy())
            rec["val"].append(((st.predict(uv) - yv) ** 2).mean(1).cpu().numpy())
            rec["fit"].append(st.predict(ug).cpu().numpy())
            for fam, ms in views.items():
                rec[f"sub_{fam}"].append(np.stack([st.predict(ug, m).cpu().numpy() for m in ms[: args.k_sub]]))
                rec[f"val_{fam}"].append(np.mean([((st.predict(uv, m) - yv) ** 2).mean(1).cpu().numpy() for m in ms], 0))
        if t < args.steps:
            st.step(u, y, args.lr * 0.5 * (1 + math.cos(math.pi * t / args.steps)))
    rec = {k: np.array(v) for k, v in rec.items()}
    rec.update(cfgs=cfgs, u=u.cpu().numpy()[:, 0], y=y.cpu().numpy(), ug=ug.cpu().numpy()[:, 0], device=dev)
    return rec


def draw(rec, args, frame=None):
    """Columns: training regimes. Rows 1-2: the full network (thick) and some of its sub-networks (thin) with
    random units or random blocks removed. Row 3: validation loss of the full network and of the sub-networks
    (mean over k_eval). frame=None draws the end of training."""
    i = len(rec["step"]) - 1 if frame is None else frame
    fig = plt.figure(figsize=(12, 8.1))
    # a thin third row holds the loss legend, so each legend sits directly above the rows it describes
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 0.2, 1], hspace=0.3, wspace=0.12,
                          left=0.075, right=0.99, top=0.9, bottom=0.07)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in (0, 1, 3)])
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")
    ug, truth = rec["ug"], np.sin(2 * np.pi * rec["ug"])
    losses = np.concatenate([rec["val"]] + [rec[f"val_{f}"] for f in FAMILIES])
    lo, hi = losses.min() * 0.75, losses.max() * 1.5
    pct = f"{args.p_view:.0%}"
    for c, cfg in enumerate(rec["cfgs"]):
        kind, col = cfg["kind"], COLOR[cfg["kind"]]
        for r, fam in enumerate(FAMILIES):
            ax = axes[r, c]
            for s in range(rec[f"sub_{fam}"].shape[1]):
                ax.plot(ug, rec[f"sub_{fam}"][i, s, c], color=col, lw=0.8, alpha=0.4, zorder=2)
            ax.plot(ug, truth, color="0.55", lw=1.1, ls=(0, (2, 2)), zorder=1)
            ax.plot(ug, rec["fit"][i, c], color=col, lw=2.0, zorder=3)
            ax.scatter(rec["u"], rec["y"], s=11, facecolor="white", edgecolor="black", lw=0.7, zorder=4)
            ax.set_xlim(0, 1)
            ax.set_ylim(-2.3, 2.3)
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"] if r == 1 else [])
            ax.set_yticks([-2, 0, 2])
            ax.grid(False)
            if r == 1:
                ax.set_xlabel("input", labelpad=1)
            if c == 0:
                ax.set_ylabel(f"{pct} of {fam}\nremoved")
            else:
                ax.set_yticklabels([])
        rate = {"wd": rf"$\lambda = {cfg.get('wd', 0):g}$", "dropout": rf"$p = {cfg.get('p', 0):g}$",
                "droppath": rf"$p = {cfg.get('p', 0):g}$"}.get(kind)
        axes[0, c].set_title("Trained with " + ("no regularization" if kind == "none" else
                                                 f"{TITLE[kind].lower()} ({rate})"), fontsize=10.5)

        ax = axes[2, c]
        steps = rec["step"][: i + 1]
        ax.plot(steps, rec["val"][: i + 1, c], color=col, lw=2.0)
        for fam in FAMILIES:
            ax.plot(steps, rec[f"val_{fam}"][: i + 1, c], color=col, lw=1.6, ms=5, mew=1.1, **FAM_STYLE[fam])
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, rec["step"][-1])
        ax.set_xticks(np.arange(0, rec["step"][-1] + 1, 1000), [f"{k}" for k in range(rec["step"][-1] // 1000 + 1)])
        ax.set_xlabel("training steps (thousands)")
        if c == 0:
            ax.set_ylabel("validation loss")
        else:
            ax.set_yticklabels([], minor=True)
            ax.set_yticklabels([])

    ink = "0.25"
    top = [Line2D([], [], ls="none", marker="o", ms=4.5, mfc="white", mec="black", mew=0.8),
           Line2D([], [], color="0.55", lw=1.1, ls=(0, (2, 2))),
           Line2D([], [], color=ink, lw=2.0),
           Line2D([], [], color=ink, lw=0.8, alpha=0.5)]
    fig.legend(top, ["training points", "true function", "full network", "sub-networks"], loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 0.995), columnspacing=1.6, handlelength=2.0)
    bottom = [Line2D([], [], color=ink, lw=2.0)] + [
        Line2D([], [], color=ink, lw=1.6, ms=5, mew=1.1, **{k: v for k, v in FAM_STYLE[f].items() if k != "markevery"})
        for f in FAMILIES]
    legend_ax.legend(bottom, ["full network"] + [f"sub-networks, {pct} of {fam} removed (mean of {args.k_eval})"
                                                  for fam in FAMILIES],
                     loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.75), columnspacing=1.6, handlelength=2.4)
    if frame is not None:
        fig.text(0.995, 0.995, f"step {rec['step'][i]:,}", ha="right", va="top", fontsize=9, color="0.3")
    return fig, axes


TRAIN_ARGS = ("n", "sigma", "d", "hid", "depth", "ln", "lr", "steps", "every", "wd", "p_dropout", "p_droppath",
              "k_sub", "k_eval", "p_view", "seed", "data_seed")


def save_rec(rec, args):
    """Cache a training run (CPU and GPU runs differ: they draw different random masks)."""
    meta = {k: getattr(args, k) for k in TRAIN_ARGS} | {"device": rec["device"], "cfgs": rec["cfgs"]}
    np.savez_compressed(FIG_DIR / f"{args.name}_rec.npz", meta=json.dumps(meta),
                        **{k: v for k, v in rec.items() if k not in ("cfgs", "device")})


def load_rec(args):
    z = np.load(FIG_DIR / f"{args.name}_rec.npz")
    meta = json.loads(str(z["meta"]))
    diff = {k: (meta[k], getattr(args, k)) for k in TRAIN_ARGS if meta[k] != getattr(args, k)}
    if diff:
        raise SystemExit(f"the cached run used different training settings (cached, requested): {diff}")
    rec = {k: z[k] for k in z.files if k != "meta"}
    rec.update(cfgs=meta["cfgs"], device=meta["device"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--ln", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--p-dropout", type=float, default=0.3)
    ap.add_argument("--p-droppath", type=float, default=0.3)
    ap.add_argument("--k-sub", type=int, default=6)
    ap.add_argument("--k-eval", type=int, default=16)
    ap.add_argument("--p-view", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=7)
    ap.add_argument("--name", default="toy_dynamics")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif-dpi", type=int, default=90)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--gif-stride", type=int, default=2)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-dpi", type=int, default=120)  # 1440 x 972 px
    ap.add_argument("--crf-h264", type=int, default=24)
    ap.add_argument("--crf-vp9", type=int, default=36)
    ap.add_argument("--cached", action="store_true", help="redraw from the cached training run")
    args = ap.parse_args()

    if args.cached:
        rec = load_rec(args)
    else:
        rec = train(args)
        save_rec(rec, args)
    print(f"training run from {rec['device']}")
    fig, _ = draw(rec, args)
    save(fig, args.name)
    for c, cfg in enumerate(rec["cfgs"]):
        b = int(rec["val"][:, c].argmin())
        print(f"{cfg['kind']:9s} val min {rec['val'][b, c]:.3f} @ {rec['step'][b]} | final val {rec['val'][-1, c]:.3f} "
              f"train {rec['train'][-1, c]:.3f} | sub-networks: units {rec['val_units'][-1, c]:.3f} "
              f"blocks {rec['val_blocks'][-1, c]:.3f}")

    if args.gif or args.video:
        # GIF and video show the same frames: every gif_stride-th record, then the last one held for ~2 s
        last = len(rec["step"]) - 1
        shown = list(range(0, last, args.gif_stride)) + [last]
        n_frames = len(shown) + 2 * args.fps
        dpis = ([args.gif_dpi] if args.gif else []) + ([args.video_dpi] if args.video else [])
        frames = {dpi: [] for dpi in dpis}
        for f in shown:
            fig, _ = draw(rec, args, frame=f)
            for dpi in dpis:
                fig.set_dpi(dpi)
                fig.canvas.draw()
                frames[dpi].append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
        outs = []
        if args.gif:
            images = [Image.fromarray(a).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
                      for a in frames[args.gif_dpi]]
            images += [images[-1]] * (n_frames - len(images))
            out = FIG_DIR / f"{args.name}.gif"
            images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / args.fps), loop=0,
                           optimize=True)
            outs.append(out)
        if args.video:
            outs += write_video(frames[args.video_dpi], n_frames, args)
        for out in outs:
            print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


def write_video(frames, n_frames, args):
    """The animation as a looping video: H.264 MP4 (every browser), VP9 WebM (fallback for browsers without
    H.264), and a poster image (the last frame) shown until playback starts or when autoplay is off."""
    frames = frames + [frames[-1]] * (n_frames - len(frames))
    with tempfile.TemporaryDirectory() as tmp:
        for k, a in enumerate(frames):
            Image.fromarray(a).save(f"{tmp}/{k:04d}.png")
        # BT.709 conversion and tags, so browsers decode the colours as encoded. lutyuv lifts white (Y=235) to
        # Y=245: quantization then still decodes to pure 255 instead of a faint grey box on the white page.
        vf = ("pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white,scale=out_color_matrix=bt709:out_range=tv,"
              "format=yuv420p,lutyuv=y='if(gte(val,233),245,val)'")
        src = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", f"{tmp}/%04d.png",
               "-vf", vf, "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-color_range", "tv", "-an"]
        mp4, webm = FIG_DIR / f"{args.name}.mp4", FIG_DIR / f"{args.name}.webm"
        # High profile, level 4.0: hardware decoders on older phones play it
        subprocess.run(src + ["-c:v", "libx264", "-profile:v", "high", "-level:v", "4.0", "-preset", "veryslow",
                              "-tune", "animation", "-crf", str(args.crf_h264), "-movflags", "+faststart", str(mp4)],
                       check=True)
        subprocess.run(src + ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(args.crf_vp9), "-deadline", "good",
                              "-cpu-used", "1", "-row-mt", "1", str(webm)], check=True)
    poster = FIG_DIR / f"{args.name}_poster.png"
    Image.fromarray(frames[-1]).quantize(colors=64, method=Image.Quantize.MEDIANCUT).save(poster, optimize=True)
    return [mp4, webm, poster]

if __name__ == "__main__":
    main()
