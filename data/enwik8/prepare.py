import os
import pickle
import numpy as np

# Load your enwik8 file
input_file_path = "enwik8"
with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()

print(f"length of dataset in characters: {len(data):,}")

# Get all unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

# Create character to integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


# Create the train, validation, and test splits
n = len(data)
train_data = data[: int(n * 0.9)]  # 90% for training
val_data = data[int(n * 0.9) : int(n * 0.95)]  # 5% for validation
test_data = data[int(n * 0.95) :]  # 5% for test

# Encode data
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
test_ids.tofile(os.path.join(os.path.dirname(__file__), "test.bin"))

# Save the meta information
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

# vocab size: 6,064
# train has 89,659,648 tokens
# val has 4,981,092 tokens
# test has 4,981,092 tokens