#!/usr/bin/env python3
"""
Tokenize all .txt files under an input directory tree and write binary token chunks.

The script loads a Hugging Face tokenizers JSON file, reads every .txt file under the
input directory recursively, tokenizes each line, and writes the resulting token IDs to
binary files named like veriseti0001.bin, veriseti0002.bin, etc.
Each output file is limited to at most 256MB of binary data.

Token ID storage format is chosen automatically from the tokenizer vocab size:
  - vocab_size <= 65536  -> uint16 (2 bytes, little-endian)
  - vocab_size <= 4294967296 -> uint32 (4 bytes, little-endian)

Usage:
    # İlk çalıştırma
    python tokenize_to_binary.py --max-workers 8 \
      --input-dir "/mnt/disc2/all_txt" --output-dir "H:/data/data_v2" \
      --tokenizer-path "kendi_tokenizerim.json"

    # Kesinti sonrası kaldığı yerden devam
    python tokenize_to_binary.py --resume --max-workers 8 \
      --input-dir "/mnt/disc2/all_txt" --output-dir "H:/data/data_v2" \
      --tokenizer-path "kendi_tokenizerim.json"

    # Eski sürüm %20'de kesildiyse: Ctrl+C → yeni kod + --resume
    # (.tmp_parallel altındaki 0000_*.bin dosyaları otomatik içe aktarılır)

    # Tüm temp dosyalar hazırsa sadece birleştir
    python tokenize_to_binary.py --merge-only \
      --input-dir "/mnt/disc2/all_txt" --output-dir "H:/data/data_v2" \
      --tokenizer-path "kendi_tokenizerim.json"
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local
from typing import Dict, List, Optional, Tuple

from tokenizers import Tokenizer

MAX_OUTPUT_BYTES = 256 * 1024 * 1024
CHECKPOINT_VERSION = 1
_WORKER_STATE = local()


@dataclass(frozen=True)
class TokenStorage:
    vocab_size: int
    pack_format: str
    token_size_bytes: int
    max_token_id: int

    @property
    def dtype_label(self) -> str:
        if self.token_size_bytes == 2:
            return "uint16"
        if self.token_size_bytes == 4:
            return "uint32"
        return f"{self.token_size_bytes}-byte"


def resolve_token_storage(vocab_size: int) -> TokenStorage:
    """Tokenizer vocab_size değerine göre binary yazım parametrelerini hesapla."""
    if vocab_size <= 0:
        raise ValueError(f"Geçersiz vocab_size: {vocab_size}")

    max_token_id = vocab_size - 1

    if vocab_size <= 65536:
        return TokenStorage(
            vocab_size=vocab_size,
            pack_format="<H",
            token_size_bytes=2,
            max_token_id=max_token_id,
        )

    if vocab_size <= 4294967296:
        return TokenStorage(
            vocab_size=vocab_size,
            pack_format="<I",
            token_size_bytes=4,
            max_token_id=max_token_id,
        )

    raise ValueError(
        f"vocab_size={vocab_size} desteklenmiyor. "
        "Maksimum desteklenen değer: 4294967296 (uint32)."
    )


def load_token_storage(tokenizer_path: str) -> TokenStorage:
    """Tokenizer JSON dosyasını yükleyip binary yazım parametrelerini döndür."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    return resolve_token_storage(vocab_size)


def find_txt_files(input_dir: Path) -> List[Path]:
    return sorted(path for path in input_dir.rglob("*.txt") if path.is_file())


def file_signature(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def temp_name_for_file(txt_path: Path) -> str:
    digest = hashlib.sha256(str(txt_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"{digest}.bin"


def legacy_temp_path(temp_dir: Path, idx: int, txt_path: Path) -> Path:
    """Eski sürümün temp dosya adlandırması: {sıra:04d}_{dosya_adı}.bin"""
    return temp_dir / f"{idx:04d}_{txt_path.stem}.bin"


def import_legacy_temps(
    checkpoint: "CheckpointManager",
    temp_dir: Path,
    txt_files: List[Path],
) -> int:
    """
    Eski tokenize_to_binary sürümünün bıraktığı .tmp_parallel dosyalarını checkpoint'e aktar.
    Eski adlandırma: 0000_kitap.bin, 0001_roman.bin, ... (sıralı indeks + stem)
    """
    imported = 0
    for idx, txt_path in enumerate(txt_files):
        if checkpoint.is_file_complete(txt_path, temp_dir):
            continue
        legacy_path = legacy_temp_path(temp_dir, idx, txt_path)
        if not legacy_path.is_file() or legacy_path.stat().st_size == 0:
            continue
        checkpoint.mark_file_complete(txt_path, legacy_path)
        imported += 1
    return imported


class CheckpointManager:
    """tokenize_checkpoint.json ile işlenen dosyaları ve merge durumunu takip eder."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.lock = Lock()
        self.data = self._empty_data()

    @staticmethod
    def _empty_data() -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "input_dir": None,
            "tokenizer_path": None,
            "tokenizer_mtime_ns": None,
            "prefix": None,
            "merge_complete": False,
            "updated_at": None,
            "files": {},
            "merge_order": [],
        }

    def load(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        with self.checkpoint_path.open("r", encoding="utf-8") as fh:
            self.data = json.load(fh)
        return True

    def save(self):
        self.data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.checkpoint_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        tmp_path.replace(self.checkpoint_path)

    def initialize(
        self,
        input_dir: Path,
        tokenizer_path: Path,
        prefix: str,
        reset: bool = False,
    ):
        tokenizer_mtime_ns = tokenizer_path.stat().st_mtime_ns
        if reset or not self.data.get("input_dir"):
            self.data = self._empty_data()

        self.data["input_dir"] = str(input_dir.resolve())
        self.data["tokenizer_path"] = str(tokenizer_path.resolve())
        self.data["tokenizer_mtime_ns"] = tokenizer_mtime_ns
        self.data["prefix"] = prefix
        if reset:
            self.data["merge_complete"] = False
            self.data["files"] = {}
            self.data["merge_order"] = []
        self.save()

    def validate_context(self, input_dir: Path, tokenizer_path: Path, prefix: str):
        expected = {
            "input_dir": str(input_dir.resolve()),
            "tokenizer_path": str(tokenizer_path.resolve()),
            "prefix": prefix,
        }
        for key, value in expected.items():
            stored = self.data.get(key)
            if stored and stored != value:
                raise SystemExit(
                    f"Checkpoint uyumsuz ({key}): kayıtlı={stored!r}, mevcut={value!r}. "
                    f"--force-restart ile sıfırlayın."
                )

        stored_mtime = self.data.get("tokenizer_mtime_ns")
        current_mtime = tokenizer_path.stat().st_mtime_ns
        if stored_mtime is not None and stored_mtime != current_mtime:
            raise SystemExit(
                "Tokenizer dosyası checkpoint'ten sonra değişmiş. "
                "--force-restart ile yeniden tokenize edin."
            )

    def get_file_record(self, txt_path: Path) -> Optional[dict]:
        return self.data.get("files", {}).get(str(txt_path.resolve()))

    def is_file_complete(self, txt_path: Path, temp_dir: Path) -> bool:
        record = self.get_file_record(txt_path)
        if not record:
            return False
        temp_path = temp_dir / record["temp_name"]
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            return False
        try:
            current = file_signature(txt_path)
        except OSError:
            return False
        return (
            record.get("source_mtime_ns") == current["mtime_ns"]
            and record.get("source_size") == current["size"]
        )

    def mark_file_complete(self, txt_path: Path, temp_path: Path):
        key = str(txt_path.resolve())
        record = {
            "temp_name": temp_path.name,
            "bytes_written": temp_path.stat().st_size,
            **file_signature(txt_path),
        }
        with self.lock:
            self.data.setdefault("files", {})[key] = record
            merge_order = self.data.setdefault("merge_order", [])
            if key not in merge_order:
                merge_order.append(key)
            self.save()

    def mark_merge_complete(self):
        with self.lock:
            self.data["merge_complete"] = True
            self.save()

    def ordered_temp_paths(self, temp_dir: Path) -> List[Path]:
        paths = []
        for key in self.data.get("merge_order", []):
            record = self.data.get("files", {}).get(key)
            if not record:
                continue
            temp_path = temp_dir / record["temp_name"]
            if temp_path.exists():
                paths.append(temp_path)
        return paths

    def completed_count(self) -> int:
        return len(self.data.get("files", {}))

    def is_merge_complete(self) -> bool:
        return bool(self.data.get("merge_complete"))


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

    def close(self):
        with self.lock:
            if self.handle is not None:
                self.handle.close()
                self.handle = None


def init_worker(tokenizer_path: str, pack_format: str, max_token_id: int):
    _WORKER_STATE.tokenizer = Tokenizer.from_file(tokenizer_path)
    _WORKER_STATE.pack_format = pack_format
    _WORKER_STATE.max_token_id = max_token_id


def process_text_file(txt_path: Path, output_path: Path) -> Tuple[bool, str, Optional[str]]:
    try:
        tokenizer = getattr(_WORKER_STATE, "tokenizer", None)
        pack_format = getattr(_WORKER_STATE, "pack_format", None)
        max_token_id = getattr(_WORKER_STATE, "max_token_id", None)
        if tokenizer is None or pack_format is None or max_token_id is None:
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
                        if token_id < 0 or token_id > max_token_id:
                            raise ValueError(
                                f"Token ID {token_id} vocab aralığı dışında (0-{max_token_id})"
                            )
                        out_fh.write(struct.pack(pack_format, token_id))
    except Exception as exc:
        return False, str(txt_path), str(exc)
    return True, str(txt_path), None


def merge_temp_files_ordered(
    temp_paths: List[Path],
    output_dir: Path,
    prefix: str,
    max_bytes: int = MAX_OUTPUT_BYTES,
) -> int:
    writer = ChunkWriter(output_dir, prefix, max_bytes)
    merged_bytes = 0
    try:
        for temp_path in temp_paths:
            with temp_path.open("rb") as src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    writer.write_bytes(chunk)
                    merged_bytes += len(chunk)
    finally:
        writer.close()
    return merged_bytes


def run_merge(
    checkpoint: CheckpointManager,
    temp_dir: Path,
    output_dir: Path,
    prefix: str,
    txt_files: List[Path],
    cleanup_temp: bool,
) -> int:
    missing = []
    for txt_path in txt_files:
        if not checkpoint.is_file_complete(txt_path, temp_dir):
            missing.append(txt_path)

    if missing:
        print(f"Merge için {len(missing)} dosya henüz tokenize edilmemiş.")
        for path in missing[:5]:
            print(f"  eksik: {path}")
        if len(missing) > 5:
            print(f"  ... ve {len(missing) - 5} dosya daha")
        raise SystemExit("Önce tokenize işlemini tamamlayın veya --resume ile devam edin.")

    temp_paths = checkpoint.ordered_temp_paths(temp_dir)
    if not temp_paths:
        raise SystemExit(f"Merge edilecek temp dosya bulunamadı: {temp_dir}")

    print(f"Merging {len(temp_paths)} temp file(s) into {output_dir}/{prefix}*.bin ...")
    merged_bytes = merge_temp_files_ordered(temp_paths, output_dir, prefix)
    checkpoint.mark_merge_complete()
    print(f"Merge tamamlandı: {_human_bytes(merged_bytes)} yazıldı.")

    if cleanup_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Temp dizini silindi: {temp_dir}")

    return merged_bytes


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def tokenize_to_binary(
    tokenizer_path: str,
    input_dir: Path,
    output_dir: Path,
    prefix: str = "veriseti",
    max_workers: int = 4,
    storage: Optional[TokenStorage] = None,
    checkpoint: Optional[CheckpointManager] = None,
    resume: bool = False,
    force_restart: bool = False,
    merge_only: bool = False,
    no_merge: bool = False,
    cleanup_temp: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path_obj = Path(tokenizer_path).resolve()

    if storage is None:
        storage = load_token_storage(tokenizer_path)

    if checkpoint is None:
        checkpoint = CheckpointManager(output_dir / "tokenize_checkpoint.json")

    checkpoint_exists = checkpoint.load()
    if force_restart:
        checkpoint.initialize(input_dir, tokenizer_path_obj, prefix, reset=True)
    elif resume:
        if not checkpoint_exists:
            print("Checkpoint bulunamadı; sıfırdan başlanıyor.")
            checkpoint.initialize(input_dir, tokenizer_path_obj, prefix, reset=True)
        else:
            checkpoint.validate_context(input_dir, tokenizer_path_obj, prefix)
    else:
        # Yeni çalıştırma: önceki ilerlemeyi sıfırla
        checkpoint.initialize(input_dir, tokenizer_path_obj, prefix, reset=True)

    temp_dir = output_dir / ".tmp_parallel"
    temp_dir.mkdir(parents=True, exist_ok=True)

    txt_files = find_txt_files(input_dir)
    if not txt_files:
        print(f"No .txt files found under {input_dir}")
        return 0

    if resume and not merge_only:
        imported = import_legacy_temps(checkpoint, temp_dir, txt_files)
        if imported:
            print(
                f"Eski sürüm temp dosyalarından {imported} dosya checkpoint'e aktarıldı "
                f"(tekrar işlenmeyecek)."
            )

    if merge_only:
        if not checkpoint_exists:
            raise SystemExit(f"Checkpoint bulunamadı: {checkpoint.checkpoint_path}")
        checkpoint.validate_context(input_dir, tokenizer_path_obj, prefix)
        if checkpoint.is_merge_complete():
            print("Checkpoint'e göre merge zaten tamamlanmış. --force-restart ile yeniden başlayın.")
            return checkpoint.completed_count()
        run_merge(checkpoint, temp_dir, output_dir, prefix, txt_files, cleanup_temp)
        return checkpoint.completed_count()

    if checkpoint.is_merge_complete() and resume:
        print("Tokenize ve merge zaten tamamlanmış. --force-restart ile yeniden başlayın.")
        return checkpoint.completed_count()

    tokens_per_chunk = MAX_OUTPUT_BYTES // storage.token_size_bytes
    print(
        f"Tokenizer: {tokenizer_path}\n"
        f"  vocab_size       : {storage.vocab_size}\n"
        f"  token storage    : {storage.dtype_label} "
        f"({storage.token_size_bytes} byte, {storage.pack_format})\n"
        f"  max chunk tokens : {tokens_per_chunk:,} per .bin file (256 MB limit)\n"
        f"  checkpoint       : {checkpoint.checkpoint_path}"
    )

    pending_files = []
    skipped = 0
    for txt_path in txt_files:
        if resume and checkpoint.is_file_complete(txt_path, temp_dir):
            skipped += 1
            continue
        pending_files.append(txt_path)

    requested_workers = max_workers or 4
    effective_workers = max(
        1, min(requested_workers, max(len(pending_files), 1), max(2, os.cpu_count() or 4))
    )

    print(
        f"Found {len(txt_files)} .txt files. "
        f"Skipped {skipped} (resume), pending {len(pending_files)}. "
        f"Workers: {effective_workers}"
    )

    if not pending_files:
        print("Tokenize edilecek yeni dosya yok.")
    else:
        completed = 0
        failed = 0
        with ThreadPoolExecutor(
            max_workers=effective_workers,
            initializer=init_worker,
            initargs=(tokenizer_path, storage.pack_format, storage.max_token_id),
        ) as executor:
            future_map = {}
            for txt_path in pending_files:
                temp_path = temp_dir / temp_name_for_file(txt_path)
                future = executor.submit(process_text_file, txt_path, temp_path)
                future_map[future] = (txt_path, temp_path)

            for future in as_completed(future_map):
                txt_path, temp_path = future_map[future]
                try:
                    ok, path, error = future.result()
                    if ok:
                        checkpoint.mark_file_complete(txt_path, temp_path)
                        completed += 1
                        print(f"Completed ({checkpoint.completed_count()}/{len(txt_files)}): {path}")
                    else:
                        failed += 1
                        if temp_path.exists():
                            temp_path.unlink(missing_ok=True)
                        print(f"Failed: {path} ({error})")
                except Exception as exc:
                    failed += 1
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                    print(f"Failed: {txt_path} ({exc})")

        print(f"Tokenize turu bitti: {completed} başarılı, {failed} hatalı, {skipped} atlandı.")

    if no_merge:
        print(f"Merge atlandı (--no-merge). Temp dosyalar: {temp_dir}")
        print(f"Tamamlanınca: python ... --merge-only --input-dir ... --output-dir ...")
        return checkpoint.completed_count()

    if checkpoint.completed_count() < len(txt_files):
        print(
            f"Henüz {len(txt_files) - checkpoint.completed_count()} dosya eksik. "
            f"--resume ile devam edin."
        )
        return checkpoint.completed_count()

    run_merge(checkpoint, temp_dir, output_dir, prefix, txt_files, cleanup_temp)
    print(f"Tamamlandı. {checkpoint.completed_count()}/{len(txt_files)} dosya. Çıktı: {output_dir}")
    return checkpoint.completed_count()


def main():
    parser = argparse.ArgumentParser(description="Tokenize .txt files into binary token chunks")
    parser.add_argument("--input-dir", required=True, help="Directory to recursively scan for .txt files")
    parser.add_argument("--output-dir", required=True, help="Directory where binary files will be written")
    parser.add_argument("--tokenizer-path", default="kendi_tokenizerim.json", help="Path to the tokenizer JSON file")
    parser.add_argument("--prefix", default="veriseti", help="Output file prefix")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel processing",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint JSON yolu (varsayılan: <output-dir>/tokenize_checkpoint.json)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Checkpoint'teki tamamlanmış dosyaları atla, kaldığı yerden devam et",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Checkpoint ve ilerlemeyi sıfırla, baştan başla",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Sadece .tmp_parallel dosyalarını veriseti*.bin olarak birleştir",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Tokenize sonrası merge yapma (ayrıca --merge-only ile birleştirilebilir)",
    )
    parser.add_argument(
        "--cleanup-temp",
        action="store_true",
        help="Başarılı merge sonrası .tmp_parallel dizinini sil",
    )
    args = parser.parse_args()

    if args.force_restart and args.resume:
        raise SystemExit("--force-restart ve --resume birlikte kullanılamaz.")
    if args.merge_only and args.no_merge:
        raise SystemExit("--merge-only ve --no-merge birlikte kullanılamaz.")

    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = (
        Path(args.checkpoint_file).expanduser().resolve()
        if args.checkpoint_file
        else output_dir / "tokenize_checkpoint.json"
    )

    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer file not found: {tokenizer_path}")
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    try:
        storage = load_token_storage(str(tokenizer_path))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    checkpoint = CheckpointManager(checkpoint_path)
    tokenize_to_binary(
        str(tokenizer_path),
        input_dir,
        output_dir,
        prefix=args.prefix,
        max_workers=args.max_workers,
        storage=storage,
        checkpoint=checkpoint,
        resume=args.resume,
        force_restart=args.force_restart,
        merge_only=args.merge_only,
        no_merge=args.no_merge,
        cleanup_temp=args.cleanup_temp,
    )


if __name__ == "__main__":
    main()
