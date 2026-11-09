"""Supervised fine-tuning data for the three datasets in the post.

Each `build_*_sft_dataset` returns `(train_iter, val_iter)`. Both iterators yield `(inputs, targets)` tensors of
shape (batch, T), padded to the longest example in the batch, with every prompt position (and padding) set to
PROMPT_MASK_VALUE in `targets`, so the loss covers response tokens only. The train iterator reshuffles every
epoch; the validation iterator cycles through a fixed order, so consecutive evals see consecutive slices.

    opencode  OpenCodeReasoning (split_0, HackerEarth source, up to 10 generations per problem),
              split 90/10 by problem id with the run's seed: each seed holds out different problems.
    gsm8k     GSM8K, canonical train/test split (the test split serves as validation).
    glaive    Glaive function calling v2: first-turn trajectories with responses of at least 100 tokens,
              deterministic 10K subsample, fixed 90/10 split.

The first call downloads the raw data from the Hugging Face Hub (at the revisions pinned below), tokenizes it
with the base model's tokenizer and caches a parquet file under <NANOCHAT_BASE_DIR>/sft_data_cache/.
"""

import os
import random
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir


# Hugging Face Hub datasets and the revisions used for the post.
OPENCODE_HF = ("nvidia/OpenCodeReasoning", "20a1ca19c0d050fe9057fc08339d6b370ec1c67a")
GSM8K_HF = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
GLAIVE_HF = ("glaiveai/glaive-function-calling-v2", "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac")

PROMPT_MASK_VALUE = -1  # cross_entropy(ignore_index=-1) skips these positions

OPENCODE_TEMPLATE_PROMPT = "### Problem:\n{input}\n### Solution:\n"  # response is appended raw
GSM8K_TEMPLATE_PROMPT = "### Question:\n{question}\n### Answer:\n"
# Glaive prompt keeps the dataset's native "USER:/ASSISTANT:" markers because
# the system field's function-schema JSON references them implicitly. The
# response can include <functioncall> JSON, FUNCTION RESPONSE blocks, and
# follow-up ASSISTANT text — i.e. the full first-turn trajectory.
GLAIVE_TEMPLATE_PROMPT = "{system}\nUSER: {user}\nASSISTANT: "


# =============================================================================
# OpenCodeReasoning
# =============================================================================

def _opencode_cache_path(max_seq_len: int, source: str = "hackerearth", n_per_question: int = 10):
    """Cache path keyed by max_seq_len, source, and n_per_question so changing any of those
    triggers re-tokenization rather than silently using a stale cache."""
    return os.path.join(get_base_dir(), "sft_data_cache",
                        f"opencode_{source}_n{n_per_question}_msl{max_seq_len}.parquet")


def _huggingface_cache_dir():
    """HuggingFace datasets cache under the nanochat base dir, so raw downloads live next to the
    tokenized caches instead of in ~/.cache/huggingface/."""
    return os.path.join(get_base_dir(), "sft_data_cache", "huggingface")


def _prepare_opencode_cache(tokenizer, max_seq_len: int, source: str = "hackerearth",
                              n_per_question: int = 10, cache_path: str = None, force: bool = False):
    """Download nvidia/OpenCodeReasoning, filter by source, subsample to
    `n_per_question` generations per unique question, format with the
    Problem/Solution template, tokenize, drop too-long examples, and write a
    parquet cache with columns `tokens` (list[int]) and `prompt_length` (int).

    Only split_0 is used (split_1's `input` field is empty for some sources, so
    the prompt template wouldn't have a usable problem statement). After filtering,
    examples are grouped by `id` (unique question) and `n_per_question` are kept
    per group (random pick with a fixed seed). Goal: a corpus small enough that
    overfitting without regularization is a real risk (~12k examples for
    hackerearth at n=10).

    Idempotent: if the cache already exists at `cache_path`, returns immediately
    unless force=True.
    """
    if cache_path is None:
        cache_path = _opencode_cache_path(max_seq_len, source, n_per_question)
    if os.path.exists(cache_path) and not force:
        return cache_path

    print(f"Building OpenCodeReasoning cache at {cache_path}")
    print(f"  source={source!r}, n_per_question={n_per_question}, max_seq_len={max_seq_len}")
    huggingface_cache = _huggingface_cache_dir()
    os.makedirs(huggingface_cache, exist_ok=True)
    # Redirect both the datasets cache (via cache_dir= below) AND the underlying
    # huggingface_hub cache (which holds raw downloads) into the project dir.
    # HF_HOME has to be set before `datasets` is imported.
    os.environ.setdefault("HF_HOME", huggingface_cache)
    from datasets import load_dataset

    # Use only split_0; split_1 has empty `input` fields for some sources.
    ds_dict = load_dataset(OPENCODE_HF[0], "split_0", revision=OPENCODE_HF[1], cache_dir=huggingface_cache)
    ds = ds_dict[list(ds_dict.keys())[0]]
    before = len(ds)
    ds = ds.filter(lambda x: x["source"] == source, num_proc=4)
    print(f"  split_0: {before:,} -> {len(ds):,} after source=={source!r}")

    # Group by question id and pick `n_per_question` per group.
    from collections import defaultdict
    by_id = defaultdict(list)
    for i, qid in enumerate(ds["id"]):
        by_id[qid].append(i)
    counts = sorted(len(v) for v in by_id.values())
    print(f"  unique questions: {len(by_id):,}")
    if counts:
        print(f"  generations per question: min={counts[0]}, "
              f"median={counts[len(counts)//2]}, max={counts[-1]}")
    rng_subsample = random.Random(0)
    selected = []
    for qid, indices in by_id.items():
        if n_per_question >= len(indices):
            selected.extend(indices)
        else:
            selected.extend(rng_subsample.sample(indices, n_per_question))
    selected.sort()  # stable order independent of dict iteration
    ds = ds.select(selected)
    print(f"  after picking up to {n_per_question} per question: {len(ds):,} examples")

    bos = tokenizer.get_bos_token_id()
    prompt_texts = [OPENCODE_TEMPLATE_PROMPT.format(input=row["input"]) for row in ds]
    response_texts = [row["solution"] for row in ds]
    ids_in_order = list(ds["id"])  # one id per row; rows with the same id are different generations

    print(f"  tokenizing prompts...")
    prompt_tok_lists = tokenizer.encode(prompt_texts, prepend=bos, num_threads=4)
    print(f"  tokenizing responses...")
    response_tok_lists = tokenizer.encode(response_texts, num_threads=4)

    tokens_col = []
    plen_col = []
    id_col = []
    too_long = 0
    for qid, p_toks, r_toks in zip(ids_in_order, prompt_tok_lists, response_tok_lists):
        full = list(p_toks) + list(r_toks)
        if len(full) > max_seq_len:
            too_long += 1
            continue
        if len(r_toks) < 1:  # no response to learn from
            continue
        tokens_col.append(full)
        plen_col.append(len(p_toks))
        id_col.append(qid)

    print(f"  kept {len(tokens_col):,} examples (dropped {too_long:,} for length > {max_seq_len})")

    # Length stats for sanity
    lens = [len(t) for t in tokens_col]
    plens = plen_col
    rlens = [L - P for L, P in zip(lens, plens)]
    def stats(name, vals):
        if not vals: return
        s = sorted(vals)
        print(f"  {name:>10}: median={s[len(s)//2]}  p90={s[int(0.9*len(s))]}  max={s[-1]}")
    stats("total_len", lens)
    stats("prompt_len", plens)
    stats("response_len", rlens)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    table = pa.Table.from_pydict({
        "tokens": pa.array(tokens_col, type=pa.list_(pa.int32())),
        "prompt_length": pa.array(plen_col, type=pa.int32()),
        "id": pa.array(id_col, type=pa.string()),
    })
    pq.write_table(table, cache_path)
    print(f"  wrote {cache_path}")
    return cache_path


def build_opencode_sft_dataset(tokenizer, max_seq_len: int, batch_size: int, device,
                                seed: int = 0, rank: int = 0, world_size: int = 1,
                                val_frac: float = 0.1,
                                source: str = "hackerearth", n_per_question: int = 10):
    """Real OpenCodeReasoning SFT dataset.

    Lazily downloads, filters, tokenizes, and caches on first call. Subsequent
    calls just load the parquet cache.

    Returns `(train_iter, val_iter)`. Batches are dynamically padded to the max length
    in each batch (not to a global max), saving compute when the batch is short.

    Train/val split is by **question id** (not by example), so different generations
    of the same problem can never appear in both train and val — avoids leakage where
    the model would otherwise memorize a problem at train time and look "good" on a
    different generation of the same problem at val time.

    DDP: pass `rank` and `world_size`. Each rank gets a strided slice of train
    and val (data[rank::world_size]).
    """
    cache_path = _prepare_opencode_cache(tokenizer, max_seq_len,
                                          source=source, n_per_question=n_per_question)

    table = pq.read_table(cache_path)
    tokens_list = table["tokens"].to_pylist()
    plen_list = table["prompt_length"].to_pylist()
    id_list = table["id"].to_pylist()
    n = len(tokens_list)

    # Group example indices by question id, then split unique ids 90/10.
    from collections import defaultdict
    indices_by_id = defaultdict(list)
    for i, qid in enumerate(id_list):
        indices_by_id[qid].append(i)
    unique_ids = sorted(indices_by_id.keys())  # deterministic order before shuffling
    rng_split = random.Random(seed)  # deterministic split, identical across ranks
    rng_split.shuffle(unique_ids)
    n_val_ids = max(1, int(val_frac * len(unique_ids)))
    val_ids = set(unique_ids[:n_val_ids])
    train_indices = []
    val_indices = []
    for qid, idxs in indices_by_id.items():
        (val_indices if qid in val_ids else train_indices).extend(idxs)
    # Within each split, shuffle for batch composition.
    rng_split.shuffle(train_indices)
    rng_split.shuffle(val_indices)

    train_indices = train_indices[rank::world_size]
    val_indices = val_indices[rank::world_size]
    if rank == 0:
        n_train_ids = len(unique_ids) - n_val_ids
        print(f"opencode SFT (id-split): {len(unique_ids):,} unique questions -> "
              f"{n_train_ids:,} train ids / {n_val_ids:,} val ids")
        print(f"  total examples: {n:,}; train (rank-local): {len(train_indices):,}; "
              f"val (rank-local): {len(val_indices):,}")

    def make_iter(idx_list, shuffle: bool):
        rng = random.Random(seed + rank)
        order = list(idx_list)
        nl = len(order)
        assert nl >= batch_size, (
            f"rank {rank}: only {nl} examples after sharding (world_size={world_size}) "
            f"but batch_size={batch_size}"
        )
        while True:
            if shuffle:
                rng.shuffle(order)
            for s in range(0, nl - batch_size + 1, batch_size):
                batch_idx = order[s:s + batch_size]
                batch_tokens = [tokens_list[i] for i in batch_idx]
                batch_plen = [plen_list[i] for i in batch_idx]
                max_L = max(len(t) for t in batch_tokens)
                T = max_L - 1  # input/target length

                # Pad with 0; the loss mask handles correctness regardless of pad value.
                inputs = torch.zeros((batch_size, T), dtype=torch.long)
                targets = torch.full((batch_size, T), PROMPT_MASK_VALUE, dtype=torch.long)
                for j, (toks, plen) in enumerate(zip(batch_tokens, batch_plen)):
                    L = len(toks)
                    inputs[j, :L - 1] = torch.tensor(toks[:-1], dtype=torch.long)
                    # Loss positions: i in [plen-1, L-1) so that target=tokens[i+1] is a response token.
                    if L - 1 > plen - 1 and plen >= 1:
                        targets[j, plen - 1:L - 1] = torch.tensor(toks[plen:L], dtype=torch.long)
                yield inputs.to(device), targets.to(device)

    return make_iter(train_indices, shuffle=True), make_iter(val_indices, shuffle=False)


# =============================================================================
# GSM8K
# =============================================================================

def _gsm8k_cache_path(max_seq_len: int, config: str = "main"):
    """Cache path keyed by (config, max_seq_len)."""
    return os.path.join(get_base_dir(), "sft_data_cache",
                        f"gsm8k_{config}_msl{max_seq_len}.parquet")


def _prepare_gsm8k_cache(tokenizer, max_seq_len: int, config: str = "main",
                          cache_path: str = None, force: bool = False):
    """Download GSM8K, format Q/A with the standard template, tokenize, drop
    too-long examples, and save a parquet cache.

    GSM8K is a grade-school math word problem dataset (~7.5k train, ~1.3k test).
    Each `answer` field contains the full chain-of-thought solution plus a final
    `#### <number>` line. We use the canonical train/test split (test serves as
    val in the build_* function); both splits are stored in one parquet with a
    `split` column.

    Schema: tokens (list[int32]), prompt_length (int32), id (string), split (string).

    Idempotent: if the cache exists at `cache_path`, returns immediately unless force=True.
    """
    if cache_path is None:
        cache_path = _gsm8k_cache_path(max_seq_len, config)
    if os.path.exists(cache_path) and not force:
        return cache_path

    print(f"Building GSM8K cache at {cache_path}")
    print(f"  config={config!r}, max_seq_len={max_seq_len}")
    huggingface_cache = _huggingface_cache_dir()
    os.makedirs(huggingface_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", huggingface_cache)  # see comment in _prepare_opencode_cache
    from datasets import load_dataset

    ds_dict = load_dataset(GSM8K_HF[0], config, revision=GSM8K_HF[1], cache_dir=huggingface_cache)
    print(f"  loaded splits: {list(ds_dict.keys())}")

    bos = tokenizer.get_bos_token_id()
    tokens_col, plen_col, id_col, split_col = [], [], [], []
    too_long = 0
    empty_response = 0

    for split_name in ["train", "test"]:
        if split_name not in ds_dict:
            continue
        ds = ds_dict[split_name]
        prompt_texts = [GSM8K_TEMPLATE_PROMPT.format(question=row["question"]) for row in ds]
        response_texts = [row["answer"] for row in ds]

        print(f"  tokenizing {split_name} prompts ({len(ds):,})...")
        prompt_tok_lists = tokenizer.encode(prompt_texts, prepend=bos, num_threads=4)
        print(f"  tokenizing {split_name} responses...")
        response_tok_lists = tokenizer.encode(response_texts, num_threads=4)

        kept = 0
        for i, (p_toks, r_toks) in enumerate(zip(prompt_tok_lists, response_tok_lists)):
            full = list(p_toks) + list(r_toks)
            if len(full) > max_seq_len:
                too_long += 1
                continue
            if len(r_toks) < 1:
                empty_response += 1
                continue
            tokens_col.append(full)
            plen_col.append(len(p_toks))
            id_col.append(f"{split_name}_{i}")
            split_col.append(split_name)
            kept += 1
        print(f"  {split_name}: kept {kept:,} examples")

    print(f"  total kept: {len(tokens_col):,} "
          f"(dropped {too_long:,} too long, {empty_response:,} empty response)")

    # Length stats per split
    for split_name in ["train", "test"]:
        idxs = [i for i, s in enumerate(split_col) if s == split_name]
        if not idxs:
            continue
        lens = [len(tokens_col[i]) for i in idxs]
        plens = [plen_col[i] for i in idxs]
        rlens = [lens[k] - plens[k] for k in range(len(idxs))]
        def stats(name, vals):
            s = sorted(vals)
            print(f"  [{split_name}] {name:>12}: median={s[len(s)//2]}  "
                  f"p90={s[int(0.9*len(s))]}  max={s[-1]}")
        stats("total_len", lens)
        stats("prompt_len", plens)
        stats("response_len", rlens)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    table = pa.Table.from_pydict({
        "tokens": pa.array(tokens_col, type=pa.list_(pa.int32())),
        "prompt_length": pa.array(plen_col, type=pa.int32()),
        "id": pa.array(id_col, type=pa.string()),
        "split": pa.array(split_col, type=pa.string()),
    })
    pq.write_table(table, cache_path)
    print(f"  wrote {cache_path}")
    return cache_path


def build_gsm8k_sft_dataset(tokenizer, max_seq_len: int, batch_size: int, device,
                              seed: int = 0, rank: int = 0, world_size: int = 1):
    """Real GSM8K SFT dataset.

    Uses the canonical GSM8K train/test split: train (~7.5k) is our train,
    test (~1.3k) is our val. No random partitioning — the official split is
    deterministic and matches conventional GSM8K usage.

    DDP: pass `rank` and `world_size` for stride-based sharding.
    """
    cache_path = _prepare_gsm8k_cache(tokenizer, max_seq_len)

    table = pq.read_table(cache_path)
    tokens_list = table["tokens"].to_pylist()
    plen_list = table["prompt_length"].to_pylist()
    split_list = table["split"].to_pylist()

    train_indices = [i for i, s in enumerate(split_list) if s == "train"]
    val_indices = [i for i, s in enumerate(split_list) if s == "test"]

    # Shuffle within each split deterministically (seed-controlled, identical across ranks)
    # so the rank-stride below is order-stable across runs.
    rng_split = random.Random(seed)
    rng_split.shuffle(train_indices)
    rng_split.shuffle(val_indices)

    train_indices = train_indices[rank::world_size]
    val_indices = val_indices[rank::world_size]
    if rank == 0:
        n_train_total = sum(1 for s in split_list if s == "train")
        n_val_total = sum(1 for s in split_list if s == "test")
        print(f"GSM8K SFT (canonical train/test): "
              f"train={n_train_total:,} (rank-local {len(train_indices):,}), "
              f"val={n_val_total:,} (rank-local {len(val_indices):,})")

    return _make_padded_iters(tokens_list, plen_list, train_indices, val_indices,
                                batch_size, device, seed=seed, rank=rank, world_size=world_size)


# =============================================================================
# Shared helpers
# =============================================================================

# Shared helper: given a tokens list, a prompt-length list, and disjoint
# train/val index lists into them, return (train_iter, val_iter) that yield
# dynamically-padded batches with the prompt-masked target convention.
def _make_padded_iters(tokens_list, plen_list, train_indices, val_indices,
                        batch_size, device, *, seed: int = 0, rank: int = 0,
                        world_size: int = 1):
    def make_iter(idx_list, shuffle: bool):
        rng = random.Random(seed + rank)
        order = list(idx_list)
        nl = len(order)
        assert nl >= batch_size, (
            f"rank {rank}: only {nl} examples after sharding (world_size={world_size}) "
            f"but batch_size={batch_size}"
        )
        while True:
            if shuffle:
                rng.shuffle(order)
            for s in range(0, nl - batch_size + 1, batch_size):
                batch_idx = order[s:s + batch_size]
                batch_tokens = [tokens_list[i] for i in batch_idx]
                batch_plen = [plen_list[i] for i in batch_idx]
                max_L = max(len(t) for t in batch_tokens)
                T = max_L - 1

                inputs = torch.zeros((batch_size, T), dtype=torch.long)
                targets = torch.full((batch_size, T), PROMPT_MASK_VALUE, dtype=torch.long)
                for j, (toks, plen) in enumerate(zip(batch_tokens, batch_plen)):
                    L = len(toks)
                    inputs[j, :L - 1] = torch.tensor(toks[:-1], dtype=torch.long)
                    if L - 1 > plen - 1 and plen >= 1:
                        targets[j, plen - 1:L - 1] = torch.tensor(toks[plen:L], dtype=torch.long)
                yield inputs.to(device), targets.to(device)

    return make_iter(train_indices, shuffle=True), make_iter(val_indices, shuffle=False)


def _load_split_indexed_cache(cache_path):
    """Read a parquet cache that has a `split` column ("train"/"test") and
    return (tokens_list, plen_list, train_indices, val_indices) where val
    is the canonical "test" split."""
    table = pq.read_table(cache_path)
    tokens_list = table["tokens"].to_pylist()
    plen_list = table["prompt_length"].to_pylist()
    split_list = table["split"].to_pylist()
    train_indices = [i for i, s in enumerate(split_list) if s == "train"]
    val_indices = [i for i, s in enumerate(split_list) if s == "test"]
    return tokens_list, plen_list, train_indices, val_indices


# =============================================================================
# Glaive function calling v2
# =============================================================================

GLAIVE_MIN_RESPONSE_TOKENS = 100  # filter out short refusals / chitchat
GLAIVE_TARGET_N = 10000           # subsample target after length filter
GLAIVE_VAL_FRAC = 0.10


def _glaive_cache_path(max_seq_len: int):
    return os.path.join(get_base_dir(), "sft_data_cache",
                        f"glaive_v2_minresp{GLAIVE_MIN_RESPONSE_TOKENS}_"
                        f"n{GLAIVE_TARGET_N}_msl{max_seq_len}.parquet")


def _parse_glaive_first_turn(chat: str):
    """Extract first USER message and the assistant's full first-turn trajectory.
    Returns (user, response) or (None, None) if parsing fails. The response may
    include <functioncall> JSON, FUNCTION RESPONSE blocks, and follow-up ASSISTANT
    text up to (but not including) the next USER turn."""
    import re
    m = re.search(r"USER:\s*", chat)
    if not m:
        return None, None
    after = chat[m.end():]
    nxt = re.search(r"\nUSER:\s*", after)
    block = after[:nxt.start()] if nxt else after
    msplit = re.search(r"\nASSISTANT:\s*", block)
    if not msplit:
        return None, None
    user_text = block[:msplit.start()].strip()
    asst_text = block[msplit.end():].strip()
    asst_text = re.sub(r"\s*<\|endoftext\|>\s*$", "", asst_text)
    return user_text, asst_text


def _prepare_glaive_cache(tokenizer, max_seq_len: int, cache_path: str = None,
                            force: bool = False):
    """Download glaive-function-calling-v2, parse first-turn assistant
    trajectories (which may include <functioncall> JSON and FUNCTION RESPONSE
    blocks), filter to response length >= GLAIVE_MIN_RESPONSE_TOKENS to drop
    short refusals/chitchat, deterministically subsample to GLAIVE_TARGET_N,
    then random 90/10 split into train/test (the source has only a `train`
    split, so we construct val ourselves).

    The dataset's tool-use format (JSON function-call markers, FUNCTION
    RESPONSE blocks) is rare in FineWeb-Edu — minimal pretraining overlap.

    Schema: tokens (list[int32]), prompt_length (int32), id (string), split (string).
    """
    if cache_path is None:
        cache_path = _glaive_cache_path(max_seq_len)
    if os.path.exists(cache_path) and not force:
        return cache_path

    print(f"Building Glaive cache at {cache_path}")
    print(f"  min_response_tokens={GLAIVE_MIN_RESPONSE_TOKENS}, "
          f"target_n={GLAIVE_TARGET_N}, max_seq_len={max_seq_len}")
    huggingface_cache = _huggingface_cache_dir()
    os.makedirs(huggingface_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", huggingface_cache)
    from datasets import load_dataset

    ds = load_dataset(GLAIVE_HF[0], revision=GLAIVE_HF[1],
                      cache_dir=huggingface_cache)["train"]
    print(f"  loaded train: {len(ds):,} rows")

    bos = tokenizer.get_bos_token_id()

    # First pass: parse and tokenize so we can length-filter on response tokens.
    print(f"  parsing and tokenizing all rows...")
    parsed_prompts, parsed_responses, parsed_ids = [], [], []
    parse_fail = 0
    for i, row in enumerate(ds):
        user, asst = _parse_glaive_first_turn(row["chat"])
        if not user or not asst:
            parse_fail += 1
            continue
        prompt = GLAIVE_TEMPLATE_PROMPT.format(system=row["system"], user=user)
        parsed_prompts.append(prompt)
        parsed_responses.append(asst)
        parsed_ids.append(str(i))
    print(f"  parsed {len(parsed_prompts):,}; parse_fail {parse_fail:,}")

    print(f"  tokenizing prompts...")
    prompt_tok_lists = tokenizer.encode(parsed_prompts, prepend=bos, num_threads=4)
    print(f"  tokenizing responses...")
    response_tok_lists = tokenizer.encode(parsed_responses, num_threads=4)

    # Apply filters: response>=N tokens AND total<=max_seq_len.
    filtered = []  # (id, p_toks, r_toks)
    short_response = 0
    too_long = 0
    for qid, p_toks, r_toks in zip(parsed_ids, prompt_tok_lists, response_tok_lists):
        if len(r_toks) < GLAIVE_MIN_RESPONSE_TOKENS:
            short_response += 1
            continue
        if len(p_toks) + len(r_toks) > max_seq_len:
            too_long += 1
            continue
        filtered.append((qid, list(p_toks), list(r_toks)))
    print(f"  after filters: {len(filtered):,} kept "
          f"(dropped {short_response:,} short response, {too_long:,} too long)")

    # Deterministic subsample to TARGET_N
    rng = random.Random(0)
    if len(filtered) > GLAIVE_TARGET_N:
        rng.shuffle(filtered)
        filtered = filtered[:GLAIVE_TARGET_N]
    print(f"  after subsample: {len(filtered):,}")

    # 90/10 train/test split (deterministic). Re-shuffle for split.
    rng2 = random.Random(1)  # different seed from subsample shuffle
    rng2.shuffle(filtered)
    n_val = max(1, int(GLAIVE_VAL_FRAC * len(filtered)))
    val_set = set(range(n_val))

    tokens_col, plen_col, id_col, split_col = [], [], [], []
    for k, (qid, p_toks, r_toks) in enumerate(filtered):
        full = p_toks + r_toks
        tokens_col.append(full)
        plen_col.append(len(p_toks))
        id_col.append(qid)
        split_col.append("test" if k in val_set else "train")

    n_train = sum(1 for s in split_col if s == "train")
    n_val_cnt = sum(1 for s in split_col if s == "test")
    print(f"  split: train={n_train:,}, val={n_val_cnt:,}")

    for split_name in ["train", "test"]:
        idxs = [i for i, s in enumerate(split_col) if s == split_name]
        if not idxs:
            continue
        lens = [len(tokens_col[i]) for i in idxs]
        plens = [plen_col[i] for i in idxs]
        rlens = [lens[k] - plens[k] for k in range(len(idxs))]
        def stats(name, vals):
            s = sorted(vals)
            print(f"  [{split_name}] {name:>12}: median={s[len(s)//2]}  "
                  f"p90={s[int(0.9*len(s))]}  max={s[-1]}")
        stats("total_len", lens)
        stats("prompt_len", plens)
        stats("response_len", rlens)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    table = pa.Table.from_pydict({
        "tokens": pa.array(tokens_col, type=pa.list_(pa.int32())),
        "prompt_length": pa.array(plen_col, type=pa.int32()),
        "id": pa.array(id_col, type=pa.string()),
        "split": pa.array(split_col, type=pa.string()),
    })
    pq.write_table(table, cache_path)
    print(f"  wrote {cache_path}")
    return cache_path


def build_glaive_sft_dataset(tokenizer, max_seq_len: int, batch_size: int, device,
                               seed: int = 0, rank: int = 0, world_size: int = 1):
    """Glaive function-calling v2 SFT dataset.

    Constructed val: random 90/10 split of the 10k subsample (Glaive only ships
    a `train` split). DDP: pass `rank` and `world_size` for stride sharding.
    """
    cache_path = _prepare_glaive_cache(tokenizer, max_seq_len)
    tokens_list, plen_list, train_indices, val_indices = _load_split_indexed_cache(cache_path)

    rng_split = random.Random(seed)
    rng_split.shuffle(train_indices)
    rng_split.shuffle(val_indices)
    train_indices = train_indices[rank::world_size]
    val_indices = val_indices[rank::world_size]
    if rank == 0:
        print(f"Glaive SFT (constructed 90/10 split): "
              f"train (rank-local)={len(train_indices):,}, "
              f"val (rank-local)={len(val_indices):,}")

    return _make_padded_iters(tokens_list, plen_list, train_indices, val_indices,
                                batch_size, device, seed=seed, rank=rank, world_size=world_size)
