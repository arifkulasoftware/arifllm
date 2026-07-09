#!/usr/bin/env python3
"""
Tokenize all .txt files under an input directory tree and write binary token chunks.

The script loads a Hugging Face tokenizers JSON file, reads every .txt file under the
input directory recursively, tokenizes each line, and writes the resulting token IDs to
binary files named like veriseti0001.bin, veriseti0002.bin, etc.
Each output file is limited to at most 256MB of binary data.

Usage:
    python tokenize_to_binary.py --input-dir "H:/data/all_txt" --output-dir "H:/data/out" --tokenizer-path "kendi_tokenizerim.json"
"""
import argparse
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tokenizers import Tokenizer

MAX_OUTPUT_BYTES = 256 * 1024 * 1024
TOKEN_SIZE_BYTES = 2


def find_txt_files(input_dir: Path):
    for path in sorted(input_dir.rglob("*.txt")):
        if path.is_file():
            yield path


class ChunkWriter:
    def __init__(self, output_dir: Path, prefix: str, max_bytes: int = MAX_OUTPUT_BYTES):
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.part_index = 0
        self.handle = None
        self.current_path = None
        self.bytes_written = 0
        self.lock = Lock()

    def _rotate(self):
        if self.handle is not None:
            self.handle.close()
        self.part_index += 1
        self.current_path = self.output_dir / f"{self.prefix}{self.part_index:04d}.bin"
        self.handle = self.current_path.open("wb")
        self.bytes_written = 0

    def write_token(self, token_id: int):
        payload = struct.pack("<H", token_id)
        with self.lock:
            if self.handle is None or self.bytes_written + len(payload) > self.max_bytes:
                self._rotate()
            self.handle.write(payload)
            self.bytes_written += len(payload)

    def close(self):
        with self.lock:
            if self.handle is not None:
                self.handle.close()
                self.handle = None


def process_text_file(tokenizer: Tokenizer, txt_path: Path, writer: ChunkWriter):
    try:
        with txt_path.open("r", encoding="utf-8", errors="ignore", buffering=1024 * 1024) as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue

                encoding = tokenizer.encode(text)
                token_ids = encoding.ids
                if not token_ids:
                    continue

                for token_id in token_ids:
                    writer.write_token(token_id)
    except Exception as exc:
        return False, str(txt_path), str(exc)
    return True, str(txt_path), None


def tokenize_to_binary(tokenizer: Tokenizer, input_dir: Path, output_dir: Path, prefix: str = "veriseti", max_workers: int = 4):
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(find_txt_files(input_dir))
    if not txt_files:
        print(f"No .txt files found under {input_dir}")
        return 0

    print(f"Found {len(txt_files)} .txt files. Beginning tokenization with {max_workers} workers...")

    writer = ChunkWriter(output_dir, prefix)
    completed = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_text_file, tokenizer, txt_path, writer): txt_path
                for txt_path in txt_files
            }

            for future in as_completed(future_map):
                txt_path = future_map[future]
                try:
                    ok, path, error = future.result()
                    if ok:
                        completed += 1
                        print(f"Completed: {path}")
                    else:
                        print(f"Failed: {path} ({error})")
                except Exception as exc:
                    print(f"Failed: {txt_path} ({exc})")
    finally:
        writer.close()

    print(f"Completed. Processed {completed}/{len(txt_files)} file(s). Output directory: {output_dir}")
    return completed


def main():
    parser = argparse.ArgumentParser(description="Tokenize .txt files into binary token chunks")
    parser.add_argument("--input-dir", required=True, help="Directory to recursively scan for .txt files")
    parser.add_argument("--output-dir", required=True, help="Directory where binary files will be written")
    parser.add_argument("--tokenizer-path", default="kendi_tokenizerim.json", help="Path to the tokenizer JSON file")
    parser.add_argument("--prefix", default="veriseti", help="Output file prefix")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of worker threads for parallel processing")
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer file not found: {tokenizer_path}")
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenize_to_binary(tokenizer, input_dir, output_dir, prefix=args.prefix, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
