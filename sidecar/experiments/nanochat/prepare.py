# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Prepare a tiny character-level corpus for the nanochat GPT.

Downloads the Tiny Shakespeare text (a small, license-free corpus) if absent,
builds a character vocabulary, and writes a train/val split as uint16 token
arrays to `data/`. Idempotent: re-running is a no-op once `data/` is populated.

This mirrors Karpathy's nanoGPT `prepare.py` step: it is the data half of the
program the researcher agent never needs to touch (only `train.py` is mutable).
"""

import os
import pickle
import urllib.request

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_TXT = os.path.join(DATA_DIR, "input.txt")
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(INPUT_TXT):
        print(f"downloading corpus -> {INPUT_TXT}")
        urllib.request.urlretrieve(DATA_URL, INPUT_TXT)  # noqa: S310

    with open(INPUT_TXT, "r", encoding="utf-8") as fh:
        data = fh.read()

    chars = sorted(set(data))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}

    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)

    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))

    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as fh:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": dict(enumerate(chars))}, fh)

    print(f"prepared: vocab_size={vocab_size} train={len(train_ids)} val={len(val_ids)}")


if __name__ == "__main__":
    main()
