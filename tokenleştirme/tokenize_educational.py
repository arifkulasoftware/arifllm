#!/usr/bin/env python3
"""
tokenize_educational.py — Tokenization'ı sıfırdan anlamak için saf Python örneği

HuggingFace `tokenizers` veya başka bir LLM kütüphanesi KULLANMAZ.
Yalnızca stdlib + isteğe bağlı numpy/psutil.

AMAÇ
-----
tokenize_to_binary.py neden saatler sürer ve GB'larca RAM ister?
Bu script her adımı açıkça yapar ve süre/bellek ölçer.

NEDEN YAVAŞ? (özet)
-------------------
1. VERİ BOYUTU
   300 GB metin = yüz milyarlarca byte disk I/O.
   Disk (özellikle SC1 HDD) saniyede ~12–125 MB → saatler sürer.

2. BPE ALGORİTMASI (her kelime için)
   Kelime karakterlere bölünür → 130.000 merge kuralı sırayla denenir.
   Rust (tokenizers) bunu optimize eder; saf Python döngüsü 10–100x yavaştır.

3. PYTHON STRING MALİYETİ
   Her satır: str nesnesi, split(), liste, encode → GC baskısı.
   Rust tarafında bellek daha verimli yönetilir.

4. TOKENIZER JSON YÜKLEME
   kendi_tokenizerim.json ~650k satır → json.load() tek başına
   yüzlerce MB RAM + onlarca saniye (ölçüm yapılır).

5. BINARY YAZMA
   Her token için struct.pack() döngüsü yavaş;
   numpy tobytes() daha hızlı ama yine de disk sınırı kalır.

6. MERGE AŞAMASI (asıl script)
   Tüm temp .bin dosyaları tekrar okunup birleştirilir → 2x ek I/O.

KULLANIM
--------
  # Kavramsal demo (küçük sözlük, anında biter)
  python tokenize_educational.py --demo

  # Gerçek tokenizer JSON ile küçük örnek (yavaş ama öğretici)
  python tokenize_educational.py \
    --tokenizer-json kendi_tokenizerim.json \
    --input-file ornek.txt \
    --output-file ornek.bin \
    --max-lines 500

  # Tek dosyada profil raporu
  python tokenize_educational.py \
    --tokenizer-json kendi_tokenizerim.json \
    --input-file buyuk.txt \
    --output-file cikti.bin \
    --max-mb 50

UYARI: Bu script ile 300 GB veriyi işlemek pratik DEĞİLDİR (haftalar sürebilir).
       Production için tokenize_to_binary.py (Rust tokenizers) kullanın.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import tracemalloc
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Bellek / süre yardımcıları
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    if not HAS_PSUTIL:
        return 0.0
    return psutil.Process().memory_info().rss / (1024 ** 2)


@dataclass
class PhaseStats:
    name: str
    seconds: float
    rss_before_mb: float
    rss_after_mb: float
    tracemalloc_peak_mb: float
    extra: str = ""

    def report(self) -> str:
        return (
            f"  [{self.name}] {self.seconds:.3f}s | "
            f"RSS {self.rss_before_mb:.1f} → {self.rss_after_mb:.1f} MB | "
            f"tracemalloc peak {self.tracemalloc_peak_mb:.1f} MB"
            + (f" | {self.extra}" if self.extra else "")
        )


class Profiler:
    def __init__(self):
        self.phases: List[PhaseStats] = []

    def run(self, name: str, fn, *args, **kwargs):
        tracemalloc.start()
        rss_b = rss_mb()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.phases.append(
            PhaseStats(
                name=name,
                seconds=elapsed,
                rss_before_mb=rss_b,
                rss_after_mb=rss_mb(),
                tracemalloc_peak_mb=peak / (1024 ** 2),
            )
        )
        return result

    def print_report(self):
        print("\n" + "=" * 72)
        print("PROFİL RAPORU")
        print("=" * 72)
        total = sum(p.seconds for p in self.phases)
        for p in self.phases:
            print(p.report())
        print(f"\n  Toplam ölçülen süre: {total:.3f}s")
        print(f"  Anlık RSS: {rss_mb():.1f} MB")
        print("=" * 72)


# ---------------------------------------------------------------------------
# Saf Python BPE (HuggingFace tokenizers JSON formatından)
# ---------------------------------------------------------------------------

@dataclass
class SimpleBPE:
    """
    BPE (Byte Pair Encoding) — kelimeleri alt kelime parçalarına böler.

    Eğitim aşaması (build_tokenizer.py) çok ağır: tüm corpus'taki kelime
    frekansları bellekte tutulur → 300 GB'da 100 GB+ RAM.

    Bu sınıf sadece ENCODE (çıkarım) yapar; yine de her kelime için
  merge listesinde arama yapar.
    """
    token_to_id: Dict[str, int]
    merge_ranks: Dict[Tuple[str, str], int]
    unk_id: int = 1

    @classmethod
    def from_hf_json(cls, path: Path) -> "SimpleBPE":
        """kendi_tokenizerim.json dosyasını json modülü ile yükle."""
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        model = data["model"]
        if model.get("type") != "BPE":
            raise ValueError(f"Desteklenen model: BPE, gelen: {model.get('type')}")

        token_to_id = model["vocab"]
        merges = model.get("merges", [])
        merge_ranks = {tuple(pair): rank for rank, pair in enumerate(merges)}
        unk_token = model.get("unk_token", "[UNK]")
        unk_id = token_to_id.get(unk_token, 1)

        return cls(token_to_id=token_to_id, merge_ranks=merge_ranks, unk_id=unk_id)

    @classmethod
    def demo_vocab(cls) -> "SimpleBPE":
        """Küçük Türkçe örnek sözlük — algoritmayı anlamak için."""
        token_to_id = {
            "[PAD]": 0, "[UNK]": 1,
            "ev": 10, "evler": 11, "güzel": 12, "bir": 13, "evlerden": 14,
            "e": 20, "v": 21, "l": 22, "er": 23, "den": 24,
        }
        merges = [
            ["e", "v"],
            ["ev", "ler"],
            ["ler", "den"],
        ]
        merge_ranks = {tuple(m): i for i, m in enumerate(merges)}
        return cls(token_to_id=token_to_id, merge_ranks=merge_ranks)

    def tokenize_word(self, word: str) -> List[str]:
        """Tek kelime için BPE merge döngüsü (saf Python — yavaş)."""
        symbols = list(word)
        if not symbols:
            return []

        while len(symbols) >= 2:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(p, self.merge_ranks[p]) for p in pairs if p in self.merge_ranks]
            if not ranked:
                break
            best_pair = min(ranked, key=lambda x: x[1])[0]

            merged: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    merged.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

        return symbols

    def encode_line(self, line: str, lowercase: bool = False) -> List[int]:
        """
        NFKC + boşlukla bölme (Whitespace pre_tokenizer ile uyumlu basit sürüm).
        """
        text = unicodedata.normalize("NFKC", line.strip())
        if not text:
            return []
        if lowercase:
            text = text.casefold()

        ids: List[int] = []
        for word in text.split():
            for piece in self.tokenize_word(word):
                ids.append(self.token_to_id.get(piece, self.unk_id))
        return ids


# ---------------------------------------------------------------------------
# Binary yazım (struct vs numpy)
# ---------------------------------------------------------------------------

def resolve_pack_format(vocab_size: int) -> Tuple[str, int]:
    if vocab_size <= 65536:
        return "<H", 2
    return "<I", 4


def write_ids_struct_loop(ids: List[int], pack_fmt: str, out_fh) -> int:
    """Her token için struct.pack — basit ama yavaş."""
    nbytes = 0
    for token_id in ids:
        out_fh.write(struct.pack(pack_fmt, token_id))
        nbytes += struct.calcsize(pack_fmt)
    return nbytes


def write_ids_numpy(ids: List[int], token_size: int, out_fh) -> int:
    """numpy array → bytes — struct döngüsünden daha hızlı."""
    if not HAS_NUMPY:
        raise RuntimeError("numpy yok; --write-mode struct kullanın")
    dtype = np.uint16 if token_size == 2 else np.uint32
    payload = np.asarray(ids, dtype=dtype).tobytes()
    out_fh.write(payload)
    return len(payload)


# ---------------------------------------------------------------------------
# Ana işlem döngüsü (tek dosya, satır satır — tokenize_to_binary ile aynı mantık)
# ---------------------------------------------------------------------------

@dataclass
class TokenizeResult:
    lines_read: int = 0
    bytes_read: int = 0
    tokens_written: int = 0
    bytes_written: int = 0
    words_tokenized: int = 0


def process_file_line_by_line(
    input_path: Path,
    output_path: Path,
    bpe: SimpleBPE,
    pack_fmt: str,
    token_size: int,
    write_mode: str,
    max_lines: Optional[int],
    max_bytes: Optional[int],
    lowercase: bool,
) -> TokenizeResult:
    """
    tokenize_to_binary.py'nin tek dosyalık, tek thread'li hali.

    Neden yavaş?
    - Her satır Python'da ayrı str
    - Her kelime için BPE while döngüsü
    - Her satırda disk write (buffer küçükse)
    """
    result = TokenizeResult()
    vocab_size = max(bpe.token_to_id.values()) + 1

    writer = write_ids_numpy if write_mode == "numpy" else None

    with input_path.open("r", encoding="utf-8", errors="ignore") as in_fh:
        with output_path.open("wb") as out_fh:
            for line in in_fh:
                if max_lines is not None and result.lines_read >= max_lines:
                    break

                raw = line.encode("utf-8", errors="ignore")
                if max_bytes is not None and result.bytes_read + len(raw) > max_bytes:
                    break
                result.bytes_read += len(raw)
                result.lines_read += 1

                ids = bpe.encode_line(line, lowercase=lowercase)
                result.words_tokenized += len(line.split())
                result.tokens_written += len(ids)

                if not ids:
                    continue

                if writer is not None:
                    result.bytes_written += writer(ids, token_size, out_fh)
                else:
                    result.bytes_written += write_ids_struct_loop(ids, pack_fmt, out_fh)

    return result


def run_demo():
    print("=" * 72)
    print("DEMO: BPE nasıl çalışır?")
    print("=" * 72)
    bpe = SimpleBPE.demo_vocab()
    samples = [
        "ev",
        "evler",
        "evlerden",
        "güzel bir evlerden",
    ]
    for text in samples:
        print(f"\nMetin: {text!r}")
        for word in text.split():
            pieces = bpe.tokenize_word(word)
            ids = [bpe.token_to_id.get(p, bpe.unk_id) for p in pieces]
            print(f"  kelime {word!r} → parçalar {pieces} → id'ler {ids}")

    print("\n" + "-" * 72)
    print("Büyük vocab'ta (131k) her kelime için merge listesinde arama yapılır.")
    print("300 GB × milyonlarca kelime × Python döngüsü = saatler/günler.")
    print("-" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Saf Python tokenization örneği (eğitim amaçlı)"
    )
    parser.add_argument("--demo", action="store_true", help="Küçük BPE demosu")
    parser.add_argument("--tokenizer-json", type=Path, help="HF format tokenizer JSON")
    parser.add_argument("--input-file", type=Path, help="Tokenize edilecek .txt")
    parser.add_argument("--output-file", type=Path, help="Çıktı .bin dosyası")
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--max-mb", type=float, default=None, help="Okunacak max MB")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument(
        "--write-mode",
        choices=["struct", "numpy"],
        default="numpy" if HAS_NUMPY else "struct",
        help="Binary yazım yöntemi",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.tokenizer_json or not args.input_file or not args.output_file:
        parser.error("--tokenizer-json, --input-file ve --output-file gerekli (--demo hariç)")

    if not args.input_file.exists():
        raise SystemExit(f"Dosya bulunamadı: {args.input_file}")

    max_bytes = int(args.max_mb * 1024 * 1024) if args.max_mb else None
    profiler = Profiler()

    print("Saf Python tokenization başlıyor...")
    print(f"  numpy : {HAS_NUMPY} | psutil: {HAS_PSUTIL} | write_mode: {args.write_mode}")

    # --- Faz 1: JSON yükleme (bellek zirvesi burada oluşabilir) ---
    def load_bpe():
        return SimpleBPE.from_hf_json(args.tokenizer_json)

    bpe = profiler.run("1_json_load", load_bpe)
    vocab_size = max(bpe.token_to_id.values()) + 1
    pack_fmt, token_size = resolve_pack_format(vocab_size)
    profiler.phases[-1].extra = (
        f"vocab={vocab_size:,}, merges={len(bpe.merge_ranks):,}, dtype={token_size}B/token"
    )

    # --- Faz 2: Tokenize + yaz ---
    def tokenize():
        return process_file_line_by_line(
            args.input_file,
            args.output_file,
            bpe,
            pack_fmt,
            token_size,
            args.write_mode,
            args.max_lines,
            max_bytes,
            args.lowercase,
        )

    result = profiler.run("2_tokenize_write", tokenize)
    mb_read = result.bytes_read / (1024 ** 2)
    mb_out = result.bytes_written / (1024 ** 2)
    tok_per_sec = result.tokens_written / profiler.phases[-1].seconds if profiler.phases[-1].seconds else 0
    mb_per_sec = mb_read / profiler.phases[-1].seconds if profiler.phases[-1].seconds else 0

    profiler.phases[-1].extra = (
        f"lines={result.lines_read:,}, read={mb_read:.1f}MB, "
        f"tokens={result.tokens_written:,} ({tok_per_sec:,.0f}/s), "
        f"written={mb_out:.1f}MB ({mb_per_sec:.1f}MB/s okuma)"
    )

    profiler.print_report()

    # --- 300 GB projeksiyon ---
    if mb_read > 0:
        sec = profiler.phases[-1].seconds
        projected_hours = (300 * 1024) / (mb_read / sec) / 3600 if sec > 0 else float("inf")
        print("\nKABA PROJEKSİYON (bu hızla 300 GB txt):")
        print(f"  Tokenize süresi : ~{projected_hours:.0f} saat (saf Python BPE)")
        print(f"  + merge I/O     : +%30–50 ek süre (asıl pipeline'da)")
        print(f"  Rust tokenizers : genelde 10–50x daha hızlı encode")
        print(f"  Asıl darboğaz   : SC1/HDD disk I/O (encode hızlı olsa bile)")

    print(f"\nÇıktı: {args.output_file} ({mb_out:.2f} MB)")
    print("\nNEDEN tokenize_to_binary.py GB bellek kullanır?")
    print("  • Bu script: satır satır işler → düşük bellek")
    print("  • build_tokenizer.py (BPE eğitimi): TÜM corpus frekans tablosu RAM'de")
    print("  • Eski train_from_iterator: tüm satırları bellekte tutuyordu → 110 GB OOM")
    print("  • Çok worker + 11k dosya: aynı anda birçok açık dosya/buffer")
    print("  • Merge: tüm temp .bin dosyalarını okuyup yeniden yazar")


if __name__ == "__main__":
    main()
