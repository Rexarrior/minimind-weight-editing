"""Make the tokenizer-aligned MiniMind2 baseline used in the paper."""
import argparse
import hashlib
from pathlib import Path
import struct


ALIGNED_HASHES = {
    "e0e4592d6015327dbf1a71b8572284becd59e03344a3a4bc1365932ad7efa22e":
        "5c504e6e2ecb8fd0bc425580951f0069cceabd01e794465bb26dc7ee57234cd8",
    "513693600fe4463417f00b019c1d20b5fbf6edca0983ce6145bd1680bd346b21":
        "174fcb623d929964fc932f7fc982b145cd2d2b4c40023829ba251e6433451a74",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original Small or Q4 GGUF")
    parser.add_argument("destination", type=Path, help="new path for the aligned copy")
    args = parser.parse_args()
    data = args.source.read_bytes()
    original_hash = hashlib.sha256(data).hexdigest()
    if original_hash not in ALIGNED_HASHES:
        parser.error("Input SHA-256 must match one of the two original upstream files.")
    # GGUF string key, GGUF_TYPE_STRING (8), and the five-byte value.
    key = b"tokenizer.ggml.pre"
    prefix = struct.pack("<Q", len(key)) + key + struct.pack("<IQ", 8, 5)
    needle = prefix + b"qwen2"
    if data.count(needle) != 1:
        parser.error("Expected one pre-tokenizer metadata field with value qwen2.")
    aligned = data.replace(needle, prefix + b"gpt-2", 1)
    digest = hashlib.sha256(aligned).hexdigest()
    if digest != ALIGNED_HASHES[original_hash]:
        parser.error("Aligned output does not match the recorded baseline.")
    with args.destination.open("xb") as out:
        out.write(aligned)
    print(f"{digest}  {args.destination}")


if __name__ == "__main__":
    main()
