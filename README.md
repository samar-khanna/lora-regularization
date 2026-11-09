# How to Regularize your LoRA Training Better

Code for the blog post [How to Regularize your LoRA Training Better](https://samar-khanna.github.io/blog/2026/lora-regularization/).
It pretrains two small language models with [nanochat](https://github.com/karpathy/nanochat), fine-tunes them with
LoRA under the six regularizers of the post's *regularization ladder*, and reproduces every number and figure in the post.
The tables behind the post are in `results/`, so the statistics and figures can be regenerated without a GPU.

## The ladder

Each rung changes one thing from the rung before it. Rungs are numbered as in the post everywhere in the code.

| Rung | Name | Noise during training | Mask |
|---|---|---|---|
| L0 | none | – | – |
| L1 | dropout(x) | dropout on the adapter's input: the standard `lora_dropout` of loralib and Hugging Face PEFT | one value per element |
| L2 | dropout(Ax) | dropout on the rank-r bottleneck | one value per element |
| L3 | dropout(BAx) | dropout on the adapter's output | one value per element |
| L4 | token DLP | drop the adapter's output for a whole token | one value per token |
| L5 | sequence DLP ("DLP") | drop the adapter's output for a whole sequence | one value per sequence |
| L6 | drop-path(Wx + BAx) | drop the whole attention or MLP sublayer, frozen weights included | one value per sequence |

All rungs rescale kept values by 1/(1-p) and are off at inference. Each adapter draws its own mask. L6's rate rises
linearly with depth, from 0 in the first block to p in the last. The implementation is `nanochat/lora.py` (L1-L5) and
`nanochat/gpt.py` (L6). `--noise gaussian` keeps a rung's mask shape but multiplies by g ~ N(1, p/(1-p)) instead of
keep/drop; L5 with Gaussian noise is the post's "Gaussian DLP" (Appendix G).

### DLP in your own code

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=32, alpha=32, p=0.3):
        super().__init__()
        self.base = base.requires_grad_(False)                # frozen W
        bound = 1 / base.in_features ** 0.5
        self.A = nn.Parameter(torch.empty(rank, base.in_features).uniform_(-bound, bound))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale, self.p = alpha / rank, p

    def forward(self, x):                                     # x: (batch, seq, in_features)
        delta = (x @ self.A.T) @ self.B.T                     # BAx
        if self.training and self.p > 0:                      # Drop-LoRA-Path:
            keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p   # one coin per sequence
            delta = delta * keep / (1 - self.p)
        return self.base(x) + self.scale * delta
```

For token DLP (L4), draw the mask with shape `(batch, seq, 1)` instead.

## Setup

The post's runs used Python 3.12 and PyTorch 2.11.0 with CUDA 13.0 on one NVIDIA DGX Spark (GB10 GPU, 128 GB unified
memory, aarch64). Install the PyTorch build for your CUDA version first, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # pick the index for your CUDA version
pip install -e ".[dev]"
python -m pytest -q                                                   # CPU tests of the ladder
```

Data, tokenizer, checkpoints and runs are written to `$NANOCHAT_BASE_DIR`, which defaults to `./data_and_checkpoints`.
Run every command from the repository root.

## Reproducing the post

| Step | Command | Produces | Time on one GB10 |
|---|---|---|---|
| 1 | `bash runs/pretrain.sh` | FineWeb-Edu shards, the tokenizer, the `standard` and `droppath` base models | about 10.7 h per base |
| 2 | `python -m scripts.sweep lora-hparams` | Appendix B: 96 LoRA configurations, no regularization | |
| 3 | `python -m scripts.sweep ladder` <br> `python -m scripts.sweep weight-decay` | the 930 runs of Figures 4-7 and Appendix C | per run: GSM8K ~26 min, Glaive ~66 min, OpenCodeReasoning ~83 min, with two runs sharing the GPU |
| 4 | `python -m scripts.sweep schedules` <br> `python -m scripts.sweep gaussian` <br> `python -m scripts.sweep checkpoints` | Appendix D, Appendix G, and the checkpointed reruns used by Figure 9 and Appendix F | |
| 5 | `python -m scripts.results collect` | `results/runs.csv`, `results/curves.parquet`, `results/pretraining.csv`, and the post's statistics | seconds |
| 6 | `python -m scripts.visualization.figures` <br> `python -m scripts.visualization.toy_dynamics --gif --video` | Figures 1, 3-7, A1, B1, and Figure 2 (needs ffmpeg) | minutes, CPU |
| 7 | see below | Figures 8-12, E1, F1 and the numbers quoted with them | |

Every sweep skips runs that already have a `summary.json`, so it can be stopped and restarted. `--dry-run` lists its
runs; `--datasets`, `--bases`, `--seeds`, `--rungs` and `--rates` narrow it; `--parallel N` runs N at a time (the post
used 2-5 with the CUDA MPS daemon running). A single run is one call of `scripts.lora_sft`, for example DLP on GSM8K:

```bash
python -m scripts.lora_sft --base-model-tag standard --dataset gsm8k --rung L5 --rate 0.3 \
    --lora-targets c_q,c_k,c_v,c_proj,c_fc --rank 32 --alpha 32 --lr 3e-3 --effective-batch-size 16 \
    --num-iterations 4500 --seed 0
```

**Interpretability analyses** ("Why does Drop-LoRA-Path work?"). They read the trained adapters and need a GPU.
They write to `results/interpretability/` and skip files that already exist there, so pass `--out <dir>` to
recompute.

```bash
for ds in opencode gsm8k glaive; do
  python -m scripts.interpretability.battery final $ds   # KL from the base, block tests, penalties (Figs 8, 10, 12)
  python -m scripts.interpretability.hybrids $ds         # adapters swapped between runs (Fig E1)
done
for run in data_and_checkpoints/lora_runs_ckpt/*/; do
  python -m scripts.interpretability.battery ckpt $(basename $run)   # KL and accuracy over training (Fig 9)
done
python -m scripts.interpretability.pair_terms            # pair terms over training (Fig F1)
python -m scripts.interpretability.report                # the numbers quoted in the section and Appendices E-F
python -m scripts.visualization.interpretability         # Figures 8-12, E1, F1
```

## What is in `results/`

- `runs.csv`: one row per fine-tuning run in the post (1,109 runs): its configuration (dataset, base, rung, rate,
  weight decay, schedule, noise, seed, LoRA settings) and metrics. `pass_floor` is the post's metric.
- `curves.parquet`: validation loss and smoothed training loss at every evaluation, keyed by `(runs_dir, run)`.
- `pretraining.csv`: pretraining curves of the two base models (Figure A1).
- `interpretability/`: the records behind Figures 8-12, E1 and F1 (JSON): `final/` and `ckpt/` from
  `scripts.interpretability.battery`, `pair_terms/` and `hybrids/` from the scripts of those names. Each script's
  docstring describes its fields.

`python -m scripts.results report` prints the post's numbers from these tables: the paired comparisons with DLP
(Figure 5, Appendix C), the chosen rates, the per-rate comparison (Figure 6), the gain over no regularization, the L6
cross-base test, and Appendices B, D and G. `python -m scripts.interpretability.report` prints the numbers of
"Why does Drop-LoRA-Path work?" and Appendices E-F.

## Notes

- **Metric.** Every 50 steps a run scores the next 160 validation examples, cycling through the validation set in a
  fixed order. Eight consecutive evaluations cover each validation set once, so a run's metric is its lowest mean over
  eight consecutive evaluations: its best full-pass validation loss, as with early stopping.
- **Comparisons.** Each method takes its best rate (or weight decay) separately for every dataset, base and seed. Pairs
  share the dataset, base and seed, and so the adapter initialization, data order and validation split. Tests are
  one-sided Wilcoxon signed-rank tests over the 15 pairs per base.
- **Rates.** The rate grid stops at 0.3, and every adapter-level rung was best at 0.3 in every cell.
- **Run-to-run noise.** GPU kernels are not deterministic: rerunning a configuration with the same seed changed the
  metric by 0.0013 nats on average (0.0046 at most, over the 30 checkpointed reruns).
- **Data.** GSM8K's validation set is its official test split. OpenCodeReasoning is split 90/10 by problem with the
  run's seed, so each seed holds out different problems; Glaive's split is fixed. The Hugging Face dataset revisions
  used for the post are pinned in `nanochat/sft_data.py`.
- **Base models.** `runs/pretrain.sh` writes them as `standard` and `droppath` (model tags used throughout).

## Repository layout

```
nanochat/           nanochat's model, tokenizer, data and training utilities (vendored), plus
  lora.py           LoRA and the regularization ladder
  sft_data.py       OpenCodeReasoning, GSM8K and Glaive fine-tuning data
  gpt.py            the transformer, with drop-path (used for the drop-path base and L6)
scripts/
  tok_train.py, base_train.py, base_eval.py    tokenizer, pretraining and evaluation (nanochat)
  lora_sft.py       LoRA fine-tuning with one rung of the ladder
  sweep.py          launches the post's runs (presets per experiment)
  results.py        collects runs into results/ and prints the post's statistics
  interpretability/ the analyses of "Why does Drop-LoRA-Path work?" and their numbers
  visualization/    the post's figures: figures.py, interpretability.py, and toy_dynamics.py (Figure 2)
runs/pretrain.sh    pretrains and evaluates both base models
results/            the tables and records behind the post
tests/              CPU tests of the ladder
```

## Citation

```bibtex
@misc{khanna2026lorareg,
  author = {Khanna, Samar},
  title  = {How to Regularize your {LoRA} Training Better},
  year   = {2026},
  month  = nov,
  url    = {https://samar-khanna.github.io/blog/2026/lora-regularization/},
  note   = {Blog post}
}
```

## Acknowledgements

Built on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy (MIT license); the files in `nanochat/` and
the pretraining scripts are adapted from it. Datasets: FineWeb-Edu, OpenCodeReasoning, GSM8K and Glaive function
calling v2, from the Hugging Face Hub.
