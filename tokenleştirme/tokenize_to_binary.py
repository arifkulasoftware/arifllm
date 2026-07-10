#!/usr/bin/env python3
"""
Tokenize all .txt files under an input directory tree and write binary token chunks.

The script loads a Hugging Face tokenizers JSON file, reads every .txt file under the
input directory recursively, tokenizes each line, and writes the resulting token IDs to
binary files named like veriseti0001.bin, veriseti0002.bin, etc.
Each output file is limited to at most 256MB of binary data.

Usage:
    python tokenize_to_binary.py --max-workers 8 --input-dir "F:/My App/ai/data/all_txt" --output-dir "H:/data/out" --tokenizer-path "kendi_tokenizerim.json"
    tüm data 24 saat 100Gb bellek peek
"""
import argparse
import os
import shutil
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, local

from tokenizers import Tokenizer

MAX_OUTPUT_BYTES = 256 * 1024 * 1024
TOKEN_SIZE_BYTES = 2
_WORKER_STATE = local()


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

    def write_bytes(self, payload: bytes):
        with self.lock:
            if self.handle is None or self.bytes_written + len(payload) > self.max_bytes:
                self._rotate()
            self.handle.write(payload)
            self.bytes_written += len(payload)

    def write_token(self, token_id: int):
        self.write_bytes(struct.pack("<H", token_id))

    def close(self):
        with self.lock:
            if self.handle is not None:
                self.handle.close()
                self.handle = None


def init_worker(tokenizer_path: str):
    _WORKER_STATE.tokenizer = Tokenizer.from_file(tokenizer_path)


def process_text_file(txt_path: Path, output_path: Path):
    try:
        tokenizer = getattr(_WORKER_STATE, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Tokenizer was not initialized for this worker")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as out_fh:
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
                        out_fh.write(struct.pack("<H", token_id))
    except Exception as exc:
        return False, str(txt_path), str(exc)
    return True, str(txt_path), None


def merge_temp_files(temp_dir: Path, output_dir: Path, prefix: str, max_bytes: int = MAX_OUTPUT_BYTES):
    writer = ChunkWriter(output_dir, prefix, max_bytes)
    try:
        for temp_path in sorted(temp_dir.glob("*.bin")):
            with temp_path.open("rb") as src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    writer.write_bytes(chunk)
    finally:
        writer.close()


def tokenize_to_binary(tokenizer_path: str, input_dir: Path, output_dir: Path, prefix: str = "veriseti", max_workers: int = 4):
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(find_txt_files(input_dir))
    if not txt_files:
        print(f"No .txt files found under {input_dir}")
        return 0

    requested_workers = max_workers or 4
    effective_workers = max(1, min(requested_workers, len(txt_files), max(2, os.cpu_count() or 4)))
    print(f"Found {len(txt_files)} .txt files. Beginning tokenization with {effective_workers} workers...")

    temp_dir = output_dir / ".tmp_parallel"
    temp_dir.mkdir(parents=True, exist_ok=True)
    completed = 0

    try:
        with ThreadPoolExecutor(max_workers=effective_workers, initializer=init_worker, initargs=(tokenizer_path,)) as executor:
            future_map = {
                executor.submit(process_text_file, txt_path, temp_dir / f"{idx:04d}_{txt_path.stem}.bin"): txt_path
                for idx, txt_path in enumerate(txt_files)
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
        if temp_dir.exists():
            merge_temp_files(temp_dir, output_dir, prefix)
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Completed. Processed {completed}/{len(txt_files)} file(s). Output directory: {output_dir}")
    return completed


def main():
    parser = argparse.ArgumentParser(description="Tokenize .txt files into binary token chunks")
    parser.add_argument("--input-dir", required=True, help="Directory to recursively scan for .txt files")
    parser.add_argument("--output-dir", required=True, help="Directory where binary files will be written")
    parser.add_argument("--tokenizer-path", default="kendi_tokenizerim.json", help="Path to the tokenizer JSON file")
    parser.add_argument("--prefix", default="veriseti", help="Output file prefix")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of worker threads for parallel processing (keeps memory usage lower than process-based workers)")
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer file not found: {tokenizer_path}")
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    tokenize_to_binary(str(tokenizer_path), input_dir, output_dir, prefix=args.prefix, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
