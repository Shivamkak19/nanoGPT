import os
import pickle
import numpy as np
from glob import glob


def process_file(input_file_path):
    # Create output directory based on the input filename
    base_name = os.path.splitext(os.path.basename(input_file_path))[0].replace(" ", "_")
    output_dir = f"processed_{base_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Load and read the file
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()

    print(f"\nProcessing {input_file_path}")
    print(f"Length of dataset in characters: {len(data):,}")

    # Get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size:,}")

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

    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")
    print(f"Test has {len(test_ids):,} tokens")

    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    test_ids = np.array(test_ids, dtype=np.uint16)

    # Save files in the category-specific directory
    train_ids.tofile(os.path.join(output_dir, "train.bin"))
    val_ids.tofile(os.path.join(output_dir, "val.bin"))
    test_ids.tofile(os.path.join(output_dir, "test.bin"))

    # Save the meta information
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    return vocab_size, len(train_ids), len(val_ids), len(test_ids)


def main():
    # Process all .txt files in the current directory
    txt_files = glob("*.txt")

    # Store statistics for each category
    stats = []

    for txt_file in txt_files:
        vocab_size, train_tokens, val_tokens, test_tokens = process_file(txt_file)
        stats.append(
            {
                "category": txt_file,
                "vocab_size": vocab_size,
                "train_tokens": train_tokens,
                "val_tokens": val_tokens,
                "test_tokens": test_tokens,
            }
        )

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    print(f"{'Category':<30} {'Vocab Size':>10} {'Train':>12} {'Val':>10} {'Test':>10}")
    print("-" * 80)
    for stat in stats:
        print(
            f"{stat['category']:<30} {stat['vocab_size']:>10,} {stat['train_tokens']:>12,} {stat['val_tokens']:>10,} {stat['test_tokens']:>10,}"
        )


if __name__ == "__main__":
    main()
