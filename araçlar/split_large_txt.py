#!/usr/bin/env python3
"""
Split large .txt files into equal chunks of at most 256MB.

For each .txt file larger than 256MB:
- it is split into smaller parts of <= 256MB,
- new files get a suffix like _part0001.txt, _part0002.txt, etc.,
- the original file is renamed to .txt.big (kept, not deleted).

Usage:
  python split_large_txt.py --input-dir "F:/My App/ai/data/all_txt/opus.nlpl"
"""
import argparse
import os
import shutil
from pathlib import Path

MAX_SIZE_BYTES = 256 * 1024 * 1024


def iter_txt_files(input_dir: Path):
    for path in sorted(input_dir.rglob("*.txt")):
        if path.is_file():
            yield path


def split_file(input_path: Path, output_dir: Path):
    size_bytes = input_path.stat().st_size
    if size_bytes <= MAX_SIZE_BYTES:
        return False

    print(f"Splitting {input_path} ({size_bytes / (1024 * 1024):.2f} MB)")

    with input_path.open("r", encoding="utf-8", errors="ignore") as src:
        chunk_index = 1
        chunk_bytes = 0
        chunk_lines = []

        for line in src:
            line_bytes = len(line.encode("utf-8"))
            if chunk_bytes + line_bytes > MAX_SIZE_BYTES and chunk_lines:
                part_path = output_dir / f"{input_path.stem}_part{chunk_index:04d}{input_path.suffix}"
                with part_path.open("w", encoding="utf-8") as dst:
                    dst.write("".join(chunk_lines))
                chunk_index += 1
                chunk_lines = []
                chunk_bytes = 0

            chunk_lines.append(line)
            chunk_bytes += line_bytes

        if chunk_lines:
            part_path = output_dir / f"{input_path.stem}_part{chunk_index:04d}{input_path.suffix}"
            with part_path.open("w", encoding="utf-8") as dst:
                dst.write("".join(chunk_lines))

    renamed_path = input_path.with_suffix(input_path.suffix + ".big")
    if renamed_path.exists():
        renamed_path.unlink()
    os.replace(input_path, renamed_path)
    print(f"Renamed original file to {renamed_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Split large .txt files into <=256MB chunks")
    parser.add_argument("--input-dir", required=True, help="Directory to scan recursively for .txt files")
    parser.add_argument("--output-dir", default=None, help="Directory for split parts (default: same directory as source)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    total = 0
    for path in iter_txt_files(input_dir):
        if split_file(path, output_dir):
            total += 1

    print(f"Done. Processed {total} large file(s).")


if __name__ == "__main__":
    main()
