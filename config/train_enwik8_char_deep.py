# config for training on enwik8
out_dir = "out-enwik8-deep"
eval_interval = 500
eval_iters = 200
log_interval = 10

dataset = "enwik8"
grad_clip = 1.0

# model settings
n_layer = 32
n_head = 8
n_embd = 768  # matching GPT-2 small
dropout = 0.0
block_size = 512  # context length for char-level

# training
batch_size = 64
max_iters = 100000
learning_rate = 3e-4
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 100000
min_lr = 3e-5
