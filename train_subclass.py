"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

Modified to support sequential fine-tuning on category subsets.
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

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out-enwik8-subclasses"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "out-enwik8/ckpt_2500.pt"  # Load from the pretrained enwik8 checkpoint

# data
dataset = "enwik8"
gradient_accumulation_steps = 5 * 8
batch_size = 128
block_size = 256

# model
n_layer = 12
n_head = 12
n_embd = 768

# Category information
categories = [
    "Arts and Literature",
    "Science",
    "Technology",
    "Geography",
    "History",
    "Politics and Government",
    "Economics and Business",
    "Religion and Philosophy",
    "Health and Medicine",
    "Culture and Society",
    "Sports and Recreation",
    "Education and Academia",
    "Language and Linguistics",
    "Military and Warfare",
    "Entertainment and Media",
    "Biographies",
]

# wandb logging
wandb_log = True
wandb_project = "enwik8-char"
wandb_run_name = f"gpt2-char-categories-{time.time()}"

# adamw optimizer
learning_rate = 3e-4
max_iters = 100  # iterations per category
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
decay_lr = True
warmup_iters = 10  # shorter warmup for fine-tuning
lr_decay_iters = 100  # should match max_iters
min_lr = 3e-5

# system
device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# -----------------------------------------------------------------------------
# Various initialization code
# (same as original but we'll wrap it in a function for reuse across categories)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps_adjusted = gradient_accumulation_steps // ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    gradient_accumulation_steps_adjusted = gradient_accumulation_steps

tokens_per_iter = (
    gradient_accumulation_steps_adjusted * ddp_world_size * batch_size * block_size
)
if master_process:
    print(f"Tokens per iteration: {tokens_per_iter:,}")
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
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


def get_batch(split, data_dir):
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
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, ctx, data_dir):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    # learning rate decay scheduler (cosine with warmup)
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train_on_category(category, prev_model_path, master_process):
    print(f"\nStarting training on category: {category}")

    # Setup paths
    category_safe = category.replace(" ", "_")
    data_dir = os.path.join(
        "data", dataset, "wiki_categories", "subclasses", f"processed_{category_safe}"
    )
    checkpoint_path = os.path.join(out_dir, f"ckpt_{category_safe}.pt")

    # Load model from previous checkpoint
    checkpoint = torch.load(prev_model_path, map_location=device)
    if "model_args" in checkpoint:
        model_args = checkpoint["model_args"]
    else:
        model_args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            bias=bias,
            vocab_size=None,
            dropout=dropout,
        )

    # META.PKL DATA IS ALREADY LOADED INTO CHECKPOINT ARGS
    # DURING PRETRAINING PHASE

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.to(device)
    if compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Initialize training state
    iter_num = 0
    best_val_loss = float("inf")

    # Initialize wandb
    if wandb_log and master_process:
        import wandb

        wandb.init(
            project=wandb_project,
            name=f"{wandb_run_name}-{category_safe}",
            config=config,
        )

    # Training loop
    X, Y = get_batch("train", data_dir)
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0

    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(model, ctx, data_dir)
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            if wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
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
                    print(f"saving checkpoint to {checkpoint_path}")
                    torch.save(checkpoint, checkpoint_path)

        # Forward backward update
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch("train", data_dir)
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )

        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()

    if wandb_log and master_process:
        wandb.finish()

    return checkpoint_path


def main():

    # Initialize grad scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # Train sequentially on each category
    current_checkpoint = init_from
    for category in categories:
        current_checkpoint = train_on_category(
            category, current_checkpoint, master_process
        )


if __name__ == "__main__":
    main()
