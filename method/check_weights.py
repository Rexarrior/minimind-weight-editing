"""Read-only comparison of the two published MiniMind2 GGUF checkpoints."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("r4", type=Path)
    parser.add_argument("r5", type=Path)
    args = parser.parse_args()
    if sha256(args.r4) != "bc19179531b4a923aa61205bb00291b0c152984c6936dbba6ceb3233da4def7b":
        raise ValueError("The first file must be the published R4.")
    if sha256(args.r5) != "4a46f5c645f9497f299b47f5141235a5564d8a2e3a46c6cb11e7f8bb7f896410":
        raise ValueError("The second file must be the published R5 pilot.")
    tensors = [{t.name: t.data for t in GGUFReader(path).tensors} for path in (args.r4, args.r5)]
    original, edited = tensors
    if original.keys() != edited.keys():
        raise ValueError("Tensor names differ.")
    norm_names = {f"blk.{layer}.attn_norm.weight" for layer in range(8, 16)}
    changed = {}
    for name, before in original.items():
        after = edited[name]
        count = int(np.count_nonzero(before.view(np.uint8) != after.view(np.uint8)))
        if count:
            changed[name] = count
        if name in norm_names:
            np.testing.assert_array_equal(after, before * np.float32(1.06))
    if set(changed) != norm_names | {"blk.15.ffn_down.weight"}:
        raise ValueError("Unexpected set of changed tensors.")
    if sum(changed.values()) != 1_276_633:
        raise ValueError("Unexpected changed byte count.")
    for name, count in sorted(changed.items()):
        print(f"{name}: {count:,} changed bytes")
    print("All eight RMSNorm gains match the 1.06 multiplier elementwise.")


if __name__ == "__main__":
    main()
