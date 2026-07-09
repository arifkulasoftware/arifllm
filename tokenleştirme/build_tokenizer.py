#!/usr/bin/env python3
"""
build_tokenizer.py

Recursively scans an input directory for .txt files and trains a BPE tokenizer
optimized for Turkish. Outputs the tokenizer JSON as `kendi_tokenizerim.json`
inside the given output directory (defaults to this script's folder).

Requirements: tokenizers (huggingface)

Usage example:
  python build_tokenizer.py --vocab-size 65535 --min-frequency 5 --lowercase --input-dir "H:/data/all_txt" --output-dir "." --log-file "./tokenizer_training.log" --checkpoint-file "./tokenizer_checkpoint.txt"

note : 
1) vocab_size az da verilse çokta verilse tüm veriler kullanıldığında, min-frequency 10000 ve üstü için token oluşmadı
2) data mC4 hariç --vocab-size 65535 --min-frequency 100 --lowercase --input-dir "H:/data/all_txt" --output-dir "." --log-file "./tokenizer_training.log" --checkpoint-file "./tokenizer_checkpoint.txt"

"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
#os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "18"
os.environ["MKL_NUM_THREADS"] = "18"
os.environ["OPENBLAS_NUM_THREADS"] = "18"

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting tokenizer training\n")
    log_file.flush()
    return log_file


def log_line(log_file, message: str):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{stamp}] {message}\n"
    print(text, end="")
    log_file.write(text)
    log_file.flush()


def find_txt_files(input_dir: str):
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                yield os.path.join(root, f)


def iter_training_lines(file_paths, max_files=None, max_lines=None, max_bytes=None, lowercase=True, log_file=None, checkpoint_file=None):
    processed_files = 0
    processed_lines = 0
    processed_bytes = 0
    total_files = len(file_paths)

    def show_progress(path=None, status="reading"):
        if path is None:
            message = f"Progress: {processed_files}/{total_files} files | {processed_lines} lines | {processed_bytes} bytes"
        else:
            message = f"Progress: {processed_files}/{total_files} files | {processed_lines} lines | {processed_bytes} bytes | {status}: {Path(path).name}"
        sys.stdout.write("\r" + message.ljust(220))
        sys.stdout.flush()

    for path in file_paths:
        if max_files is not None and processed_files >= max_files:
            break
        processed_files += 1

        show_progress(path=path, status="processing")
        if log_file is not None:
            log_line(log_file, f"Processing file {processed_files}/{total_files}: {path}")

        try:
            with open(path, "r", encoding="utf-8", errors="ignore", buffering=1024 * 1024) as fh:
                for line in fh:
                    if max_lines is not None and processed_lines >= max_lines:
                        show_progress(path=path, status="limit reached")
                        if log_file is not None:
                            log_line(log_file, f"Reached max_lines limit at {processed_lines} lines")
                        return

                    s = line.strip()
                    if not s:
                        continue

                    if lowercase:
                        s = s.casefold()

                    if max_bytes is not None:
                        line_bytes = len(s.encode("utf-8"))
                        if processed_bytes + line_bytes > max_bytes:
                            show_progress(path=path, status="size limit reached")
                            if log_file is not None:
                                log_line(log_file, f"Reached max_bytes limit at {processed_bytes} bytes")
                            return
                        processed_bytes += line_bytes

                    processed_lines += 1
                    yield s
        except Exception as exc:
            show_progress(path=path, status=f"error: {exc}")
            if log_file is not None:
                log_line(log_file, f"Error while processing {path}: {exc}")
                log_line(log_file, traceback.format_exc())
            continue

        if checkpoint_file is not None:
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file.write_text(
                f"processed_files={processed_files}\nprocessed_lines={processed_lines}\nprocessed_bytes={processed_bytes}\nlast_file={path}\n",
                encoding="utf-8",
            )

    show_progress(path=None, status="done")
    if log_file is not None:
        log_line(log_file, f"Finished reading data. Total lines: {processed_lines}, bytes: {processed_bytes}")
    print()


def build_and_save_tokenizer(file_iterator, output_path: str, vocab_size: int = 32000, min_frequency: int = 1000, log_file=None):
    if log_file is not None:
        log_line(log_file, f"Starting tokenizer training with vocab_size={vocab_size}, min_frequency={min_frequency}")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[SOS]", "[EOS]"]
    )

    try:
        tokenizer.train_from_iterator(file_iterator, trainer=trainer)
        tokenizer.save(output_path)
        if log_file is not None:
            log_line(log_file, f"Tokenizer saved successfully to {output_path}")
    except Exception as exc:
        if log_file is not None:
            log_line(log_file, f"Tokenizer training failed: {exc}")
            log_line(log_file, traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description="Train a Turkish-optimized tokenizer from .txt files")
    parser.add_argument("--input-dir", required=True, help="Input directory to recursively search for .txt files")
    parser.add_argument("--output-dir", default=".", help="Directory to write kendi_tokenizerim.json")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=1000)
    parser.add_argument("--max-files", type=int, default=None, help="Limit the number of .txt files used for training")
    parser.add_argument("--max-lines", type=int, default=None, help="Limit the number of text lines used for training")
    parser.add_argument("--max-bytes", type=int, default=None, help="Limit the total training bytes (approximate)")
    parser.add_argument("--lowercase", action="store_true", help="Apply casefold() to each line before training")
    parser.add_argument("--log-file", default="tokenizer_training.log", help="Path to the log file")
    parser.add_argument("--checkpoint-file", default=None, help="Optional file to save the last processed file and counters")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, "kendi_tokenizerim.json")
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = Path(output_dir) / log_path
    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else None
    if checkpoint_path is not None and not checkpoint_path.is_absolute():
        checkpoint_path = Path(output_dir) / checkpoint_path
    log_file = setup_logger(log_path)

    files = list(find_txt_files(input_dir))
    if not files:
        log_line(log_file, f"No .txt files found under {input_dir}")
        log_file.close()
        return

    log_line(log_file, f"Found {len(files)} .txt files. Beginning tokenizer training...")
    log_line(log_file, "Training limits: max_files=%s max_lines=%s max_bytes=%s lowercase=%s" % (
        args.max_files, args.max_lines, args.max_bytes, args.lowercase
    ))
    iterator = iter_training_lines(
        files,
        max_files=args.max_files,
        max_lines=args.max_lines,
        max_bytes=args.max_bytes,
        lowercase=args.lowercase,
        log_file=log_file,
        checkpoint_file=checkpoint_path,
    )
    build_and_save_tokenizer(iterator, out_path, vocab_size=args.vocab_size, min_frequency=args.min_frequency, log_file=log_file)
    log_line(log_file, f"Tokenizer saved to: {out_path}")
    log_file.close()


if __name__ == "__main__":
    main()
