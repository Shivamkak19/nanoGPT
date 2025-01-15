"""
Training script modified to support TIPA (Token Internal Position Awareness).
Can be run on single GPU or with DDP.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_tipa import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values for enwik8 with TIPA
out_dir = "out-enwik8"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# data
dataset = "enwik8"
gradient_accumulation_steps = 5 * 8
batch_size = 128
block_size = 256

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
tipa_enabled = True  # Enable TIPA by default

# wandb logging
wandb_log = True
wandb_project = "enwik8-char"
wandb_run_name = f"gpt2-char-tipa-l{n_layer}-h{n_head}-e{n_embd}-{time.time()}"

# optimizer
learning_rate = 3e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP settings
backend = "nccl"
device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True

# -----------------------------------------------------------------------------
# parse config overrides
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Various initialization
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# Setup directories and seeds
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"

# Data type setup
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# Data loader
data_dir = os.path.join("data", dataset)


def get_batch_with_tipa(split):
    """Get a batch of data with TIPA reverse position information"""
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    # Create reverse position tensors
    reverse_positions = torch.stack(
        [torch.arange(block_size, 0, -1, dtype=torch.long) for _ in range(batch_size)]
    )

    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        reverse_positions = reverse_positions.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
        reverse_positions = reverse_positions.to(device)

    return x, y, reverse_positions


# Model initialization
iter_num = 0
best_val_loss = 1e9

# Load meta.pkl for vocabulary size
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model initialization
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    tipa_enabled=tipa_enabled,
)

if init_from == "scratch":
    print("Initializing a new model from scratch")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "bias",
        "vocab_size",
        "tipa_enabled",
    ]:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

model.to(device)

# Initialize GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# Optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

# Compile the model if requested
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# DDP wrap if needed
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, reverse_pos = get_batch_with_tipa(split)
            with ctx:
                logits, loss = model(X, Y, reverse_pos)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# Initialize wandb
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training loop
X, Y, reverse_pos = get_batch_with_tipa("train")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # Learning rate update
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluation and logging
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        train_bpc = losses["train"] / math.log(2)
        val_bpc = losses["val"] / math.log(2)
        train_ppl = torch.exp(torch.tensor(losses["train"]))
        val_ppl = torch.exp(torch.tensor(losses["val"]))

        print(f"step {iter_num}:")
        print(f"train: {train_bpc:.4f} bpc, perplexity {train_ppl:.4f}")
        print(f"val:   {val_bpc:.4f} bpc, perplexity {val_ppl:.4f}")

        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss_bpc": train_bpc,
                    "val/loss_bpc": val_bpc,
                    "train/perplexity": train_ppl,
                    "val/perplexity": val_ppl,
                    "lr": lr,
                    "mfu": running_mfu * 100,
                }
            )

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

    # Forward and backward passes
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )

        with ctx:
            logits, loss = model(X, Y, reverse_pos)
            loss = loss / gradient_accumulation_steps

        # Prefetch next batch
        X, Y, reverse_pos = get_batch_with_tipa("train")

        # Backward pass
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
