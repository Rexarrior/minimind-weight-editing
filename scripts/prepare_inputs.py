"""Build the original likelihood scorer's binary inputs from locally downloaded public sources."""
import argparse
import hashlib
import json
import os
import struct
import tempfile
import zipfile
from pathlib import Path

from tokenizers import Tokenizer

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
SOURCE_NAMES = {"tinystories": "tiny.txt", "blimp": "blimp.zip", "copa": "test.parquet"}
EXPECTED = {
    "tinystories": "78af0a749da59eec211d542e674f507a419461ee24e19af7e9f968a550639452",
    "blimp": "e9c2c2450b18942b0de91b25438d4117f36a34e2f3dcf7e2d8d06b9602405d53",
    "copa": "3b8e50d7ac2cd912521eaae62c2857eab32bc685624966c074af362caddbc2f8",
}
TOKENIZER_HASH = "d98595c6aef70d95f72748582fb9b4f53d76dd58c1ae1dd702ad7c84e1caf5e4"


def verify(path, expected):
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"{path.name}: expected {expected}, got {observed}")


def records(task, path):
    if task == "tinystories":
        for piece in path.read_text(encoding="utf-8").split("<|endoftext|>"):
            story = piece.strip()
            if story:
                yield "", story
    elif task == "blimp":
        with zipfile.ZipFile(path) as archive:
            names = sorted(n for n in archive.namelist() if n.startswith("data/") and n.endswith(".jsonl"))
            for name in names:
                for line in archive.read(name).splitlines():
                    pair = json.loads(line)
                    yield "", pair["sentence_good"]
                    yield "", pair["sentence_bad"]
    else:
        import pyarrow.parquet as parquet
        for item in parquet.ParquetFile(path).read().to_pylist():
            prefix = item["premise"].strip()[:-1] + {"cause": " because", "effect": " therefore"}[item["question"]]
            for key in ("choice1", "choice2"):
                choice = item[key]
                yield prefix, prefix + " " + choice[:1].lower() + choice[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=SOURCE_NAMES)
    parser.add_argument("source", type=Path, help="benchmark file identified by its hash in dataset-sources.json")
    parser.add_argument("tokenizer", type=Path, help="author's 6,400-token tokenizer.json")
    parser.add_argument("output", type=Path, help="new scorer input binary")
    args = parser.parse_args()
    sources = json.loads((EVIDENCE / "dataset-sources.json").read_text())
    verify(args.source, sources[SOURCE_NAMES[args.task]]["sha256"])
    verify(args.tokenizer, TOKENIZER_HASH)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as staging:
        binary = Path(staging) / "corpus.bin"
        with binary.open("wb") as target:
            for prefix, text in records(args.task, args.source):
                sequence = [1] + tokenizer.encode(text, add_special_tokens=False).ids
                start = [1] + tokenizer.encode(prefix, add_special_tokens=False).ids
                if sequence[:len(start)] != start or len(sequence) <= len(start):
                    raise ValueError("Invalid teacher-forcing prefix")
                target.write(struct.pack("<ii", len(sequence), len(start)))
                target.write(struct.pack(f"<{len(sequence)}i", *sequence))
        verify(binary, EXPECTED[args.task])
        os.link(binary, args.output)
    print(f"{args.task}: input matches the measured corpus, SHA-256 {EXPECTED[args.task]}")


if __name__ == "__main__":
    main()
