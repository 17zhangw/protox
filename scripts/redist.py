import argparse
from pathlib import Path
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Prog")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--num-parts", type=int, required=True)
    args = parser.parse_args()

    inputs = [f for f in args.input.glob("embeddings*")]

    for i in range(args.num_parts):
        (args.input / f"part{i}").mkdir(parents=True, exist_ok=True)

    for i, emb in enumerate(inputs):
        part = f"part{i % args.num_parts}"
        shutil.move(emb, args.input / part)
