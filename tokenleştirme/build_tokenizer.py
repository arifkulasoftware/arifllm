#!/usr/bin/env python3
"""
build_tokenizer.py

Türkçe veriler için tokenizer oluşturma işlemini yapar, input_dir ile belirtilen kaynak
ve altındaki kaynaklar reküsiv olarak taranır ve .txt dosyalarından beilrtilen vocab_size için tokenlar
BPE tokenizer kullanılarak oluşturulur.
sonuç ön tanımlı olarak kendi_tokenizerim.json dosyasına yazılır.

Requirements: tokenizers (huggingface), psutil

Usage example:
  python build_tokenizer.py --vocab-size 131072 --min-frequency 5 --lowercase --input-dir "F:/My App/ai/data/all_txt" --output-dir "." --log-file "./tokenizer_training.log" --checkpoint-file "./tokenizer_checkpoint.txt"
  python build_tokenizer.py \
  --vocab-size 131072 \
  --min-frequency 25 \
  --lowercase \
  --input-dir "/mnt/disc2/all_txt" \
  --output-dir "." \
  --log-file "./tokenizer_training.log" \
  --memory-log-interval 30

notlar
* 300 Gb txt veri için çalıştığında 65335 vocab_size için 90GB bellek kullanımı oluştu
* min-frequency 10000 ve üstü için token oluşmadı
* mC4 hariç --vocab-size 65535 --min-frequency 100 --lowercase oluştu
* 4.5Gb txt veri. (sadece kitaplar , archive.org,epubs,pdf) --vocab-size 131072 --min-frequency 2 --lowercase -> 10dk işlem süresi ve vocab_size=129090 oldu.

"""
import argparse
import atexit
import faulthandler
import os
import platform
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "14"
os.environ["MKL_NUM_THREADS"] = "14"
os.environ["OPENBLAS_NUM_THREADS"] = "14"

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

try:
    import psutil
except ImportError:
    psutil = None

_ACTIVE_LOG_FILE = None
_MEMORY_MONITOR_STOP = threading.Event()


def _bytes_to_human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def format_memory_stats(label: str = "") -> str:
    """İşlem ve sistem bellek kullanımını okunabilir metin olarak döndür."""
    prefix = f"{label} | " if label else ""
    if psutil is None:
        return f"{prefix}psutil yüklü değil; RAM bilgisi alınamadı (pip install psutil)"

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()

    parts = [
        prefix + "RAM",
        f"process_rss={_bytes_to_human(mem_info.rss)}",
        f"process_vms={_bytes_to_human(mem_info.vms)}",
    ]
    if hasattr(mem_info, "data"):
        parts.append(f"process_data={_bytes_to_human(mem_info.data)}")

    parts.extend([
        f"system_total={_bytes_to_human(system_mem.total)}",
        f"system_available={_bytes_to_human(system_mem.available)}",
        f"system_used={system_mem.percent:.1f}%",
    ])
    return " | ".join(parts)


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


def log_memory(log_file, label: str):
    log_line(log_file, format_memory_stats(label))


def log_environment(log_file):
    try:
        import tokenizers as tokenizers_pkg
        tokenizers_version = getattr(tokenizers_pkg, "__version__", "unknown")
    except Exception:
        tokenizers_version = "unknown"

    log_line(
        log_file,
        "Ortam: "
        f"python={sys.version.split()[0]} | "
        f"platform={platform.platform()} | "
        f"pid={os.getpid()} | "
        f"tokenizers={tokenizers_version} | "
        f"psutil={getattr(psutil, '__version__', 'not installed')}",
    )
    log_memory(log_file, "Başlangıç")


def _write_fatal_log(message: str):
    global _ACTIVE_LOG_FILE
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{stamp}] FATAL: {message}\n"
    sys.stderr.write(text)
    if _ACTIVE_LOG_FILE is not None:
        try:
            _ACTIVE_LOG_FILE.write(text)
            _ACTIVE_LOG_FILE.flush()
        except Exception:
            pass


def install_crash_handlers(log_file):
    """Yakalanmamış hatalar, sinyaller ve native crash'ler için log yaz."""
    global _ACTIVE_LOG_FILE
    _ACTIVE_LOG_FILE = log_file

    try:
        faulthandler.enable(file=log_file, all_threads=True)
        log_line(log_file, "faulthandler etkin (native crash dump log dosyasına yazılır)")
    except Exception as exc:
        log_line(log_file, f"faulthandler etkinleştirilemedi: {exc}")

    def excepthook(exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            _write_fatal_log("KeyboardInterrupt ile sonlandırıldı")
            log_memory(log_file, "KeyboardInterrupt anı")
        else:
            tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            _write_fatal_log(f"Yakalanmamış istisna:\n{tb_text}")
            log_memory(log_file, "Yakalanmamış istisna anı")
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = excepthook

    def on_exit():
        if _MEMORY_MONITOR_STOP.is_set():
            return
        try:
            log_line(log_file, f"İşlem sonlandı. {format_memory_stats('Çıkış')}")
            log_file.flush()
        except Exception:
            pass

    atexit.register(on_exit)

    def signal_handler(signum, frame):
        signame = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        _write_fatal_log(f"Sinyal alındı: {signame} (signum={signum})")
        log_memory(log_file, f"Sinyal {signame}")
        log_file.flush()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, signal_handler)
        except (AttributeError, ValueError, OSError):
            pass

    if hasattr(signal, "SIGABRT"):
        try:
            signal.signal(signal.SIGABRT, signal_handler)
        except (ValueError, OSError):
            pass


class MemoryMonitor:
    """Arka planda periyodik RAM kullanımını loglar (OOM öncesi son durumu yakalamak için)."""

    def __init__(self, log_file, interval_sec: int = 60, phase_label: str = "monitor"):
        self.log_file = log_file
        self.interval_sec = interval_sec
        self.phase_label = phase_label
        self._thread = None

    def start(self):
        global _MEMORY_MONITOR_STOP
        _MEMORY_MONITOR_STOP.clear()
        self._thread = threading.Thread(target=self._run, name="memory-monitor", daemon=True)
        self._thread.start()
        log_line(
            self.log_file,
            f"Bellek izleme başladı: phase={self.phase_label}, interval={self.interval_sec}s",
        )

    def stop(self):
        global _MEMORY_MONITOR_STOP
        _MEMORY_MONITOR_STOP.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 5)
            self._thread = None
        log_line(self.log_file, f"Bellek izleme durdu: phase={self.phase_label}")

    def _run(self):
        tick = 0
        while not _MEMORY_MONITOR_STOP.wait(self.interval_sec):
            tick += 1
            try:
                log_memory(self.log_file, f"{self.phase_label} tick={tick}")
            except Exception as exc:
                _write_fatal_log(f"Bellek izleme hatası: {exc}")


def find_txt_files(input_dir: str):
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                yield os.path.join(root, f)


def iter_training_lines(
    file_paths,
    max_files=None,
    max_lines=None,
    max_bytes=None,
    lowercase=True,
    log_file=None,
    checkpoint_file=None,
    memory_log_every_files=500,
):
    processed_files = 0
    processed_lines = 0
    processed_bytes = 0
    total_files = len(file_paths)

    def show_progress(path=None, status="reading"):
        if path is None:
            message = f"Progress: {processed_files}/{total_files} files | {processed_lines} lines | {processed_bytes} bytes"
        else:
            message = (
                f"Progress: {processed_files}/{total_files} files | {processed_lines} lines | "
                f"{processed_bytes} bytes | {status}: {Path(path).name}"
            )
        sys.stdout.write("\r" + message.ljust(220))
        sys.stdout.flush()

    for path in file_paths:
        if max_files is not None and processed_files >= max_files:
            break
        processed_files += 1

        show_progress(path=path, status="processing")
        if log_file is not None:
            log_line(log_file, f"Processing file {processed_files}/{total_files}: {path}")
            if memory_log_every_files > 0 and processed_files % memory_log_every_files == 0:
                log_memory(log_file, f"Dosya okuma ilerlemesi ({processed_files}/{total_files})")

        try:
            with open(path, "r", encoding="utf-8", errors="ignore", buffering=1024 * 1024) as fh:
                for line in fh:
                    if max_lines is not None and processed_lines >= max_lines:
                        show_progress(path=path, status="limit reached")
                        if log_file is not None:
                            log_line(log_file, f"Reached max_lines limit at {processed_lines} lines")
                            log_memory(log_file, "max_lines limiti")
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
                                log_memory(log_file, "max_bytes limiti")
                            return
                        processed_bytes += line_bytes

                    processed_lines += 1
                    yield s
        except Exception as exc:
            show_progress(path=path, status=f"error: {exc}")
            if log_file is not None:
                log_line(log_file, f"Error while processing {path}: {type(exc).__name__}: {exc}")
                log_line(log_file, traceback.format_exc())
                log_memory(log_file, f"Dosya okuma hatası ({path})")
            continue

        if checkpoint_file is not None:
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file.write_text(
                f"processed_files={processed_files}\nprocessed_lines={processed_lines}\n"
                f"processed_bytes={processed_bytes}\nlast_file={path}\n",
                encoding="utf-8",
            )

    show_progress(path=None, status="done")
    if log_file is not None:
        log_line(log_file, f"Finished reading data. Total lines: {processed_lines}, bytes: {processed_bytes}")
        log_memory(log_file, "Dosya okuma tamamlandı (iterator henüz tüketilmedi)")
    print()


def build_and_save_tokenizer(
    file_iterator,
    output_path: str,
    vocab_size: int = 32000,
    min_frequency: int = 1000,
    log_file=None,
    memory_log_interval: int = 60,
):
    monitor = None
    if log_file is not None:
        log_line(log_file, f"Starting tokenizer training with vocab_size={vocab_size}, min_frequency={min_frequency}")
        log_memory(log_file, "BPE eğitimi öncesi")
        log_line(
            log_file,
            "NOT: train_from_iterator çağrısı sırasında tokenizers kütüphanesi veriyi bellekte "
            "işler; bu aşamada RAM kullanımı hızla artabilir (OOM killer Python traceback üretmez).",
        )

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[SOS]", "[EOS]"],
    )

    try:
        if log_file is not None and memory_log_interval > 0:
            monitor = MemoryMonitor(log_file, interval_sec=memory_log_interval, phase_label="BPE eğitimi")
            monitor.start()

        if log_file is not None:
            log_line(log_file, "train_from_iterator başlıyor...")
            log_file.flush()

        tokenizer.train_from_iterator(file_iterator, trainer=trainer)

        if log_file is not None:
            log_memory(log_file, "BPE eğitimi tamamlandı")
            log_line(log_file, f"Tokenizer kaydediliyor: {output_path}")

        tokenizer.save(output_path)

        if log_file is not None:
            log_line(log_file, f"Tokenizer saved successfully to {output_path}")
            log_memory(log_file, "Kayıt sonrası")
    except MemoryError as exc:
        if log_file is not None:
            log_line(log_file, f"Tokenizer training failed (MemoryError): {exc}")
            log_line(log_file, traceback.format_exc())
            log_memory(log_file, "MemoryError anı")
        raise
    except Exception as exc:
        if log_file is not None:
            log_line(log_file, f"Tokenizer training failed ({type(exc).__name__}): {exc}")
            log_line(log_file, traceback.format_exc())
            log_memory(log_file, f"Hata anı ({type(exc).__name__})")
        raise
    finally:
        if monitor is not None:
            monitor.stop()


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
    parser.add_argument(
        "--memory-log-interval",
        type=int,
        default=60,
        help="BPE eğitimi sırasında RAM log aralığı (saniye). 0 = sadece kilometre taşlarında logla.",
    )
    parser.add_argument(
        "--memory-log-every-files",
        type=int,
        default=500,
        help="Dosya okuma sırasında kaç dosyada bir RAM loglansın. 0 = kapalı.",
    )
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
    install_crash_handlers(log_file)

    exit_code = 0
    try:
        if psutil is None:
            log_line(log_file, "UYARI: psutil yüklü değil. RAM logları atlanır. Kurulum: pip install psutil")

        log_environment(log_file)
        log_line(
            log_file,
            f"Parametreler: input_dir={input_dir} output_dir={output_dir} vocab_size={args.vocab_size} "
            f"min_frequency={args.min_frequency} lowercase={args.lowercase} "
            f"memory_log_interval={args.memory_log_interval}s memory_log_every_files={args.memory_log_every_files}",
        )

        files = list(find_txt_files(input_dir))
        if not files:
            log_line(log_file, f"No .txt files found under {input_dir}")
            return

        log_line(log_file, f"Found {len(files)} .txt files. Beginning tokenizer training...")
        log_memory(log_file, "Dosya listesi hazır")
        log_line(
            log_file,
            "Training limits: max_files=%s max_lines=%s max_bytes=%s lowercase=%s"
            % (args.max_files, args.max_lines, args.max_bytes, args.lowercase),
        )

        iterator = iter_training_lines(
            files,
            max_files=args.max_files,
            max_lines=args.max_lines,
            max_bytes=args.max_bytes,
            lowercase=args.lowercase,
            log_file=log_file,
            checkpoint_file=checkpoint_path,
            memory_log_every_files=args.memory_log_every_files,
        )
        build_and_save_tokenizer(
            iterator,
            out_path,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            log_file=log_file,
            memory_log_interval=args.memory_log_interval,
        )
        log_line(log_file, f"Tokenizer saved to: {out_path}")
        log_memory(log_file, "Başarılı tamamlanma")
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
        log_line(log_file, f"SystemExit: code={exc.code}")
        log_memory(log_file, "SystemExit")
        raise
    except BaseException as exc:
        exit_code = 1
        log_line(log_file, f"main() başarısız ({type(exc).__name__}): {exc}")
        log_line(log_file, traceback.format_exc())
        log_memory(log_file, f"main() hata ({type(exc).__name__})")
        raise
    finally:
        global _MEMORY_MONITOR_STOP
        _MEMORY_MONITOR_STOP.set()
        try:
            log_line(log_file, f"Log kapatılıyor (exit_code={exit_code}). {format_memory_stats('Final')}")
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main() or 0)
