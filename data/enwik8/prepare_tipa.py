import os
import pickle
import numpy as np
import json
from typing import Dict, List, Tuple


class Enwik8TIPA:
    def __init__(self, input_file_path: str = "enwik8"):
        self.input_file_path = input_file_path
        self.chars = None
        self.stoi = None
        self.itos = None
        self.vocab_size = None

    def create_char_mappings(self, data: str) -> None:
        """Create character-to-integer and integer-to-character mappings."""
        self.chars = sorted(list(set(data)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def create_reverse_mapping(self, sequence: str) -> Dict[int, str]:
        """Create reverse position mapping for a sequence of characters.

        Args:
            sequence: Input string of characters

        Returns:
            Dictionary mapping positions (counting from end) to characters
        """
        return {len(sequence) - i: c for i, c in enumerate(sequence)}

    def encode(self, s: str) -> List[int]:
        """Encode string to list of integers."""
        return [self.stoi[c] for c in s]

    def decode(self, l: List[int]) -> str:
        """Decode list of integers to string."""
        return "".join([self.itos[i] for i in l])

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare enwik8 data with reverse position mapping.

        Returns:
            Tuple of (train_data, val_data, test_data) as numpy arrays
        """
        # Read input file
        with open(self.input_file_path, "r", encoding="utf-8") as f:
            data = f.read()

        print(f"Length of dataset in characters: {len(data):,}")

        # Create character mappings if not already created
        if self.chars is None:
            self.create_char_mappings(data)

        print(f"Vocabulary size: {self.vocab_size:,}")

        # Create splits
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) : int(n * 0.95)]
        test_data = data[int(n * 0.95) :]

        # Create TIPA mappings
        train_mappings = []
        window_size = 256  # Same as block_size in config

        # Create sliding window reverse mappings for training data
        for i in range(0, len(train_data) - window_size + 1, window_size):
            window = train_data[i : i + window_size]
            reverse_map = self.create_reverse_mapping(window)
            train_mappings.append(reverse_map)

        # Save metadata
        meta = {
            "vocab_size": self.vocab_size,
            "itos": self.itos,
            "stoi": self.stoi,
            "reverse_mappings": train_mappings,
        }

        # Save meta information
        meta_path = os.path.join(os.path.dirname(self.input_file_path), "tipa/meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        # Encode data
        train_ids = np.array(self.encode(train_data), dtype=np.uint16)
        val_ids = np.array(self.encode(val_data), dtype=np.uint16)
        test_ids = np.array(self.encode(test_data), dtype=np.uint16)

        # Save encoded data
        train_ids.tofile(
            os.path.join(os.path.dirname(self.input_file_path), "tipa/train.bin")
        )
        val_ids.tofile(
            os.path.join(os.path.dirname(self.input_file_path), "tipa/val.bin")
        )
        test_ids.tofile(
            os.path.join(os.path.dirname(self.input_file_path), "tipa/test.bin")
        )

        print(f"Train has {len(train_ids):,} tokens")
        print(f"Val has {len(val_ids):,} tokens")
        print(f"Test has {len(test_ids):,} tokens")

        return train_ids, val_ids, test_ids


if __name__ == "__main__":
    processor = Enwik8TIPA()
    processor.prepare_data()
