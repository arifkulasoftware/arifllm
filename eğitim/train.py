#!/usr/bin/env python3
"""
LLM Model Training Script

Bu script:
- kendi_tokenizerim.json dosyasını yükler
- Binary dosyalarından (veriseti*.bin) eğitim verilerini okur
- PyTorch kullanarak LLM modeli eğitir
- Eğitilen modeli kaydeder

Büyük veri seti özellikleri (151GB+ için):
- Chunk tabanlı veri yükleme (RAM'e tüm veriyi yüklemez)
- Gradient Accumulation (bellek artırmadan büyük batch simülasyonu)
- AMP / bfloat16 mixed precision (VRAM ~%30 azalır)
- save_every_steps: N adımda bir otomatik checkpoint
- Warmup LR scheduler
- Periyodik loss loglama (log_every_steps)
- Checkpoint'ten devam (--resume)

Kullanım (Config dosyası ile - önerilen):
    python train.py --config config-rtx4060_200m.yaml

Kullanım (Config + CLI override):
    python train.py --config config-rtx4060_200m.yaml --batch-size 2

Kullanım (Checkpoint'ten devam):
    python train.py --config config-rtx4060_200m.yaml --resume H:/data/models/model_rtx4060_200m.pt

"""

import argparse
import logging
import logging.handlers
import sys
import traceback
import math
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler
from tokenizers import Tokenizer
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None

try:
    import numpy as np
except ImportError:
    np = None


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path = None) -> logging.Logger:
    """
    Logger'ı yapılandır.

    Hem konsola hem de (log_path verilmişse) dönen log dosyasına yazar.
    Log dosyası 10 MB dolduğunda yeni bir dosyaya geçer; 5 yedek tutar.

    Args:
        log_path: Log dosyasının yolu. None ise sadece konsola yazar.

    Returns:
        Yapılandırılmış logger nesnesi.
    """
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Konsol handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Dosya handler (isteğe bağlı)
    if log_path is not None:
        log_path = Path(log_path).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info(f"Log dosyası: {log_path}")

    return logger


# ---------------------------------------------------------------------------
# Dataset: Chunk tabanlı (büyük veri setleri için)
# ---------------------------------------------------------------------------

class ChunkedBinaryTokenDataset(IterableDataset):
    """
    151GB+ binary token dosyalarından chunk parça yükleyerek okur.

    Tüm veriyi tek seferde RAM'e yüklemek yerine:
      - Dosyaları chunk'lara böler (varsayılan: 10 GB/chunk)
      - Her chunk'ı sırayla işler, sonra belleği serbest bırakır

    Bu şekilde herhangi bir boyuttaki veri seti çalışır.
    """

    def __init__(
        self,
        data_dir: Path,
        data_pattern: str,
        context_length: int = 512,
        vocab_size: int = None,
        chunk_size_gb: float = 10.0,
        shuffle_files: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.data_pattern = data_pattern
        self.context_length = context_length
        self.vocab_size = vocab_size
        # uint16 → 2 byte/token
        self.chunk_size_tokens = int(chunk_size_gb * 1024 ** 3 / 2)
        self.shuffle_files = shuffle_files

        self.bin_files: List[Path] = sorted(self.data_dir.glob(data_pattern))
        if not self.bin_files:
            raise FileNotFoundError(
                f"Binary dosya bulunamadı: {self.data_dir}/{data_pattern}"
            )

        _log = logging.getLogger("train")
        total_bytes = sum(f.stat().st_size for f in self.bin_files)
        total_gb = total_bytes / 1024 ** 3
        _log.info(
            f"Bulunan {len(self.bin_files)} binary dosya | "
            f"Toplam boyut: {total_gb:.1f} GB | "
            f"Chunk boyutu: {chunk_size_gb:.1f} GB"
        )

    def _iter_chunks(self) -> Iterator[np.ndarray]:
        """
        Dosyaları sırayla okur ve chunk_size_tokens büyüklüğünde
        numpy dizileri olarak verir.
        """
        files = list(self.bin_files)
        if self.shuffle_files:
            import random
            random.shuffle(files)

        buffer = np.array([], dtype=np.uint16)

        for bin_file in files:
            _log = logging.getLogger("train")
            size_gb = bin_file.stat().st_size / 1024 ** 3
            _log.info(f"Okunuyor: {bin_file.name}  ({size_gb:.2f} GB)")
            try:
                file_tokens = np.fromfile(bin_file, dtype=np.uint16)
            except Exception as exc:
                _log.error(f"Dosya okunurken hata: {bin_file}  →  {exc}")
                continue

            buffer = np.concatenate([buffer, file_tokens])
            del file_tokens

            # Tam chunk'ları ver, kalanı sonraki dosya ile birleştir
            while len(buffer) >= self.chunk_size_tokens:
                yield buffer[: self.chunk_size_tokens]
                buffer = buffer[self.chunk_size_tokens :]

        # Son kalan parçayı ver (chunk'tan küçük olabilir)
        if len(buffer) > self.context_length + 1:
            yield buffer

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Chunk'ları sırayla işleyerek (input, target) çiftleri döndürür.
        Her chunk işlendikten sonra bellek serbest bırakılır.
        """
        for chunk in self._iter_chunks():
            num_examples = len(chunk) // (self.context_length + 1)

            for i in range(num_examples):
                start = i * self.context_length
                end = start + self.context_length

                input_tokens = chunk[start:end]
                target_token = chunk[end]

                # vocab_size sınır kontrolü
                if self.vocab_size is not None:
                    if (input_tokens >= self.vocab_size).any():
                        continue
                    if target_token >= self.vocab_size:
                        continue

                yield (
                    torch.tensor(input_tokens, dtype=torch.long),
                    torch.tensor(target_token, dtype=torch.long),
                )

            del chunk  # chunk belleği serbest bırak


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleTransformer(nn.Module):
    """Basit Transformer tabanlı LLM modeli"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        context_length: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length

        # Embedding katmanları
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(context_length, embedding_dim)

        # Transformer encoder katmanları
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Çıkış katmanı
        self.output_head = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Ağırlık paylaşımı (Weight Tying)
        if tie_weights:
            self.output_head.weight = self.token_embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Şekli (batch_size, context_length) olan token ID'leri

        Returns:
            Şekli (batch_size, vocab_size) olan logitler
        """
        seq_length = x.size(1)

        token_emb = self.token_embedding(x)           # (batch, seq_len, embedding_dim)

        positions = torch.arange(
            seq_length, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        pos_emb = self.positional_embedding(positions)  # (1, seq_len, embedding_dim)

        embeddings = self.dropout(token_emb + pos_emb)
        transformer_output = self.transformer_encoder(embeddings)
        last_token_output = transformer_output[:, -1, :]
        logits = self.output_head(last_token_output)   # (batch, vocab_size)

        return logits


# ---------------------------------------------------------------------------
# LR Scheduler: Linear Warmup + Cosine Decay
# ---------------------------------------------------------------------------

def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Warmup → Cosine decay LR scheduler.

    - warmup_steps adımda linear olarak max LR'ye ulaşır.
    - Sonra cosine ile min_lr_ratio * max_lr'ye iner.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint yardımcıları
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    global_step: int,
    epoch: int,
    loss: float,
    logger: logging.Logger,
) -> None:
    """Checkpoint dosyasını kaydet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)
    logger.info(f"💾 Checkpoint kaydedildi: {path}  (step={global_step}, loss={loss:.4f})")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[int, int, float]:
    """
    Checkpoint'i yükle.

    Returns:
        (global_step, start_epoch, best_loss)
    """
    logger.info(f"📂 Checkpoint yükleniyor: {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    global_step = ckpt.get("global_step", 0)
    start_epoch = ckpt.get("epoch", 0) + 1
    best_loss = ckpt.get("loss", float("inf"))

    logger.info(
        f"✓ Checkpoint yüklendi  |  global_step={global_step}  "
        f"epoch={start_epoch - 1}  loss={best_loss:.4f}"
    )
    return global_step, start_epoch, best_loss


# ---------------------------------------------------------------------------
# Config yükleme
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """YAML config dosyasını yükle"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")

    if yaml is None:
        raise ImportError(
            "PyYAML kütüphanesi yüklenemedi. "
            "Lütfen şunu çalıştırın: pip install pyyaml"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Geçersiz config dosyası: {config_path}")

    return config


def _set_if_none(args: argparse.Namespace, attr: str, value: Any) -> None:
    """args.<attr> None ise value ile ata."""
    if not hasattr(args, attr) or getattr(args, attr) is None:
        setattr(args, attr, value)


def merge_config_and_args(
    config: Dict[str, Any], args: argparse.Namespace
) -> argparse.Namespace:
    """
    Config dosyasından parametreleri yükle ve merge et.
    Komut satırı parametreleri config dosyasını override eder.
    """
    _set_if_none(args, "tokenizer_path", config.get("tokenizer_path"))
    _set_if_none(args, "data_dir",       config.get("data_dir"))

    # data.*
    data_cfg = config.get("data", {})
    _set_if_none(args, "data_pattern", data_cfg.get("pattern"))

    # model.*
    model_cfg = config.get("model", {})
    _set_if_none(args, "embedding_dim",  model_cfg.get("embedding_dim"))
    _set_if_none(args, "num_layers",     model_cfg.get("num_layers"))
    _set_if_none(args, "num_heads",      model_cfg.get("num_heads"))
    _set_if_none(args, "ff_dim",         model_cfg.get("ff_dim"))
    _set_if_none(args, "context_length", model_cfg.get("context_length"))
    _set_if_none(args, "dropout",        model_cfg.get("dropout"))
    _set_if_none(args, "tie_weights",    model_cfg.get("tie_weights"))

    # training.*
    train_cfg = config.get("training", {})
    _set_if_none(args, "epochs",                      train_cfg.get("epochs"))
    _set_if_none(args, "batch_size",                  train_cfg.get("batch_size"))
    _set_if_none(args, "gradient_accumulation_steps", train_cfg.get("gradient_accumulation_steps"))
    _set_if_none(args, "learning_rate",               train_cfg.get("learning_rate"))
    _set_if_none(args, "warmup_steps",                train_cfg.get("warmup_steps"))
    _set_if_none(args, "weight_decay",                train_cfg.get("weight_decay"))
    _set_if_none(args, "max_grad_norm",               train_cfg.get("max_grad_norm"))
    _set_if_none(args, "device",                      train_cfg.get("device"))
    _set_if_none(args, "use_amp",                     train_cfg.get("use_amp"))
    _set_if_none(args, "amp_dtype",                   train_cfg.get("amp_dtype"))

    # checkpoint.*
    ckpt_cfg = config.get("checkpoint", {})
    _set_if_none(args, "model_output_path", ckpt_cfg.get("model_output_path"))
    _set_if_none(args, "save_every_steps",  ckpt_cfg.get("save_every_steps"))
    _set_if_none(args, "resume",            ckpt_cfg.get("resume"))

    # logging.*
    log_cfg = config.get("logging", {})
    _set_if_none(args, "log_path",        log_cfg.get("log_path"))
    _set_if_none(args, "log_every_steps", log_cfg.get("log_every_steps"))

    # chunk (büyük veri seti için)
    _set_if_none(args, "chunk_size_gb", config.get("chunk_size_gb"))

    return args


# ---------------------------------------------------------------------------
# Eğitim döngüsü
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    args: argparse.Namespace,
    logger: logging.Logger,
    model_output_path: Path,
    start_global_step: int = 0,
    start_epoch: int = 1,
    best_loss: float = float("inf"),
) -> None:
    """
    Ana eğitim döngüsü.

    Özellikler:
    - Gradient accumulation (gradient_accumulation_steps)
    - AMP mixed precision (use_amp + scaler)
    - save_every_steps: N global adımda bir checkpoint
    - log_every_steps: N adımda bir loss logu
    - Warmup + cosine LR scheduler
    - KeyboardInterrupt anında güvenli kayıt
    """
    grad_accum = getattr(args, "gradient_accumulation_steps", 1) or 1
    save_every = getattr(args, "save_every_steps", 0) or 0
    log_every  = getattr(args, "log_every_steps",  100) or 100
    use_amp    = getattr(args, "use_amp", False) or False
    max_grad_norm = getattr(args, "max_grad_norm", 1.0) or 1.0

    amp_dtype = torch.bfloat16
    if getattr(args, "amp_dtype", "bfloat16") == "float16":
        amp_dtype = torch.float16

    global_step = start_global_step
    accum_loss  = 0.0
    accum_count = 0

    # Checkpoint için ara dosya yolu: <model>_ckpt_step_<N>.pt
    def ckpt_path(step: int) -> Path:
        stem = model_output_path.stem
        return model_output_path.parent / f"{stem}_ckpt_step_{step}.pt"

    logger.info(
        f"Eğitim başlıyor  |  epochs={args.epochs}  "
        f"batch={args.batch_size}  grad_accum={grad_accum}  "
        f"efektif_batch={args.batch_size * grad_accum}  "
        f"AMP={use_amp}  dtype={amp_dtype}"
    )
    logger.info("-" * 80)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f"═══ Epoch {epoch}/{args.epochs} başladı ═══")
            model.train()
            optimizer.zero_grad()

            epoch_loss   = 0.0
            epoch_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)

            for batch_idx, (input_ids, target_ids) in enumerate(pbar):
                input_ids  = input_ids.to(device, non_blocking=True)
                target_ids = target_ids.to(device, non_blocking=True)

                # ----- Forward pass (AMP) -----
                if use_amp and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(input_ids)
                        loss   = criterion(logits, target_ids)
                        loss   = loss / grad_accum
                    scaler.scale(loss).backward()
                else:
                    logits = model(input_ids)
                    loss   = criterion(logits, target_ids) / grad_accum
                    loss.backward()

                raw_loss     = loss.item() * grad_accum
                accum_loss  += raw_loss
                accum_count += 1
                epoch_loss  += raw_loss
                epoch_batches += 1

                # ----- Optimizer step (gradient accumulation) -----
                is_accum_step = ((batch_idx + 1) % grad_accum == 0)

                if is_accum_step:
                    if use_amp and device.type == "cuda":
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    optimizer.zero_grad()
                    global_step += 1

                    # ----- Loglama -----
                    if log_every and global_step % log_every == 0:
                        avg = accum_loss / max(accum_count, 1)
                        lr  = optimizer.param_groups[0]["lr"]
                        logger.info(
                            f"step={global_step:>8d}  epoch={epoch}  "
                            f"loss={avg:.4f}  lr={lr:.2e}"
                        )
                        accum_loss  = 0.0
                        accum_count = 0

                    # ----- Adım başına checkpoint -----
                    if save_every and global_step % save_every == 0:
                        cp = ckpt_path(global_step)
                        avg_loss_now = epoch_loss / max(epoch_batches, 1)
                        save_checkpoint(
                            cp, model, optimizer, scheduler, scaler,
                            global_step, epoch, avg_loss_now, logger,
                        )
                        # En iyi modeli güncelle
                        if avg_loss_now < best_loss:
                            best_loss = avg_loss_now
                            torch.save(model.state_dict(), model_output_path)
                            logger.info(
                                f"⭐ Yeni en iyi model: {model_output_path}  "
                                f"(loss={best_loss:.4f})"
                            )

                    pbar.set_postfix({"loss": f"{raw_loss:.4f}", "step": global_step})

            # ----- Kalan gradyanları uygula (son batch tamamlanmamış olabilir) -----
            if (batch_idx + 1) % grad_accum != 0:
                if use_amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # ----- Epoch sonu -----
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            logger.info(
                f"═══ Epoch {epoch}/{args.epochs} tamamlandı  "
                f"|  Ortalama Loss: {avg_epoch_loss:.4f}  "
                f"|  Global Step: {global_step} ═══"
            )

            # Epoch checkpoint
            ep_ckpt = model_output_path.parent / f"{model_output_path.stem}_epoch_{epoch}.pt"
            save_checkpoint(
                ep_ckpt, model, optimizer, scheduler, scaler,
                global_step, epoch, avg_epoch_loss, logger,
            )

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), model_output_path)
                logger.info(
                    f"⭐ Yeni en iyi model: {model_output_path}  "
                    f"(loss={best_loss:.4f})"
                )

    except KeyboardInterrupt:
        logger.warning("⚠ Eğitim kullanıcı tarafından durduruldu (Ctrl+C).")
        logger.warning("  Son model kaydediliyor...")
        interrupt_path = model_output_path.parent / f"{model_output_path.stem}_interrupted.pt"
        save_checkpoint(
            interrupt_path, model, optimizer, scheduler, scaler,
            global_step, epoch if "epoch" in dir() else start_epoch,
            accum_loss / max(accum_count, 1), logger,
        )
        logger.info(f"  Interrupt checkpoint: {interrupt_path}")
        logger.info("  Devam etmek için: --resume " + str(interrupt_path))

    except Exception:
        logger.error("❌ Eğitim sırasında beklenmeyen hata!")
        logger.error(traceback.format_exc())
        raise

    logger.info("-" * 80)
    logger.info(f"Eğitim tamamlandı!  En iyi loss: {best_loss:.4f}")
    logger.info(f"Model: {model_output_path}")


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def get_vocab_size(tokenizer_path: Path) -> int:
    """Tokenizer'dan vocab boyutunu al"""
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer.get_vocab_size()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM Model Eğitim Script (büyük veri seti desteği)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:

  1. Config dosyası ile (önerilen):
    python train.py --config config-rtx4060_200m.yaml

  2. Config + CLI override:
    python train.py --config config-rtx4060_200m.yaml --batch-size 2

  3. Checkpoint'ten devam:
    python train.py --config config-rtx4060_200m.yaml \\
      --resume H:/data/models/model_rtx4060_200m_ckpt_step_5000.pt

  4. OOM alırsan:
    python train.py --config config-rtx4060_200m.yaml --batch-size 2 --context-length 64
        """,
    )

    # Config
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config dosyasının yolu")

    # Paths
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--data-dir",       default=None)
    parser.add_argument("--data-pattern",   default=None,
                        help="Binary dosya glob pattern (default: veriseti*.bin)")
    parser.add_argument("--model-output-path", default=None)

    # Model
    parser.add_argument("--embedding-dim",  type=int,   default=None)
    parser.add_argument("--num-layers",     type=int,   default=None)
    parser.add_argument("--num-heads",      type=int,   default=None)
    parser.add_argument("--ff-dim",         type=int,   default=None)
    parser.add_argument("--context-length", type=int,   default=None)
    parser.add_argument("--dropout",        type=float, default=None)
    parser.add_argument(
        "--tie-weights",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=None,
    )

    # Training
    parser.add_argument("--epochs",                      type=int,   default=None)
    parser.add_argument("--batch-size",                  type=int,   default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int,   default=None,
                        help="Gradient accumulation adım sayısı (efektif batch = batch*steps)")
    parser.add_argument("--learning-rate",               type=float, default=None)
    parser.add_argument("--warmup-steps",                type=int,   default=None,
                        help="LR warmup adım sayısı")
    parser.add_argument("--weight-decay",                type=float, default=None)
    parser.add_argument("--max-grad-norm",               type=float, default=None,
                        help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--device",  choices=["cuda", "cpu"], default=None)
    parser.add_argument("--use-amp", type=lambda x: str(x).lower() in ["true", "1", "yes"],
                        default=None, help="Mixed precision AMP (bfloat16/float16)")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default=None)

    # Checkpoint
    parser.add_argument("--resume",           default=None,
                        help="Devam edilecek checkpoint (.pt) dosyası")
    parser.add_argument("--save-every-steps", type=int, default=None,
                        help="N adımda bir checkpoint kaydet (0 = kapalı)")

    # Logging
    parser.add_argument("--log-path",        default=None)
    parser.add_argument("--log-every-steps", type=int, default=None,
                        help="N adımda bir loss logu yaz (default: 100)")

    # Veri chunk'lama
    parser.add_argument("--chunk-size-gb", type=float, default=None,
                        help="Bellekten bir seferde okunacak veri miktarı GB (default: 10.0)")

    args = parser.parse_args()

    # ------ Config merge ------
    if args.config:
        config = load_config(args.config)
        args   = merge_config_and_args(config, args)

    # ------ Logger ------
    logger = setup_logger(getattr(args, "log_path", None))
    if args.config:
        logger.info(f"Config yüklendi: {args.config}")

    # ------ Zorunlu parametre kontrolleri ------
    if not getattr(args, "tokenizer_path", None):
        parser.error("--tokenizer-path gerekli")
    if not getattr(args, "data_dir", None):
        parser.error("--data-dir gerekli")
    if not getattr(args, "model_output_path", None):
        parser.error("--model-output-path gerekli")

    # ------ Varsayılan değerler ------
    defaults = {
        "data_pattern":              "veriseti*.bin",
        "embedding_dim":             256,
        "num_layers":                4,
        "num_heads":                 8,
        "ff_dim":                    1024,
        "context_length":            512,
        "dropout":                   0.1,
        "tie_weights":               True,
        "epochs":                    1,
        "batch_size":                4,
        "gradient_accumulation_steps": 8,
        "learning_rate":             3e-5,
        "warmup_steps":              1000,
        "weight_decay":              0.1,
        "max_grad_norm":             1.0,
        "use_amp":                   True,
        "amp_dtype":                 "bfloat16",
        "save_every_steps":          5000,
        "log_every_steps":           100,
        "chunk_size_gb":             10.0,
    }
    for attr, val in defaults.items():
        if not hasattr(args, attr) or getattr(args, attr) is None:
            setattr(args, attr, val)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------ Path nesneleri ------
    tokenizer_path    = Path(args.tokenizer_path).expanduser().resolve()
    data_dir          = Path(args.data_dir).expanduser().resolve()
    model_output_path = Path(args.model_output_path).expanduser().resolve()

    if not tokenizer_path.exists():
        logger.error(f"Tokenizer bulunamadı: {tokenizer_path}")
        sys.exit(1)
    if not data_dir.exists():
        logger.error(f"Veri dizini bulunamadı: {data_dir}")
        sys.exit(1)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------ Cihaz ------
    device = torch.device(args.device)
    logger.info(f"Cihaz: {device}")
    if device.type == "cuda":
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # ------ Vocab ------
    logger.info(f"Tokenizer yükleniyor: {tokenizer_path}")
    vocab_size = get_vocab_size(tokenizer_path)
    logger.info(f"Vocab boyutu: {vocab_size:,}")

    # ------ Dataset & DataLoader ------
    logger.info(f"Dataset oluşturuluyor: {data_dir}/{args.data_pattern}")
    dataset = ChunkedBinaryTokenDataset(
        data_dir=data_dir,
        data_pattern=args.data_pattern,
        context_length=args.context_length,
        vocab_size=vocab_size,
        chunk_size_gb=args.chunk_size_gb,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,   # IterableDataset ile 0 worker güvenli
        pin_memory=(device.type == "cuda"),
    )

    # ------ Model ------
    logger.info("Model oluşturuluyor...")
    model = SimpleTransformer(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        context_length=args.context_length,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
    ).to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Toplam parametre:      {total_params:,}  (~{total_params/1e6:.1f}M)")
    logger.info(f"Eğitilebilir parametre:{trainable_params:,}")

    # ------ Optimizer ------
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # ------ Scheduler ------
    # total_steps tahmini (veri boyutu bilinmediği için warmup_steps × 100 kullan)
    estimated_total_steps = max(args.warmup_steps * 100, 100_000)
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=estimated_total_steps,
    )

    # ------ AMP Scaler ------
    scaler = None
    if args.use_amp and device.type == "cuda":
        # bfloat16 için scaler gerekmez; float16 için gerekir
        if args.amp_dtype == "float16":
            scaler = GradScaler()
            logger.info("AMP: float16 + GradScaler aktif")
        else:
            logger.info("AMP: bfloat16 aktif (scaler gerekmez)")
    else:
        logger.info("AMP: kapalı (fp32)")

    criterion = nn.CrossEntropyLoss()

    # ------ Resume ------
    global_step = 0
    start_epoch = 1
    best_loss   = float("inf")

    if getattr(args, "resume", None):
        resume_path = Path(args.resume).expanduser().resolve()
        if resume_path.exists():
            global_step, start_epoch, best_loss = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, device, logger
            )
        else:
            logger.error(f"Resume checkpoint bulunamadı: {resume_path}")
            sys.exit(1)

    # ------ Eğitim ------
    logger.info("=" * 80)
    logger.info(f"  Eğitim Parametreleri")
    logger.info(f"  epochs            : {args.epochs}")
    logger.info(f"  batch_size        : {args.batch_size}")
    logger.info(f"  grad_accum_steps  : {args.gradient_accumulation_steps}")
    logger.info(f"  efektif batch     : {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  learning_rate     : {args.learning_rate}")
    logger.info(f"  warmup_steps      : {args.warmup_steps}")
    logger.info(f"  save_every_steps  : {args.save_every_steps}")
    logger.info(f"  log_every_steps   : {args.log_every_steps}")
    logger.info(f"  chunk_size_gb     : {args.chunk_size_gb}")
    logger.info(f"  context_length    : {args.context_length}")
    logger.info(f"  use_amp           : {args.use_amp}  ({args.amp_dtype})")
    logger.info("=" * 80)

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        logger=logger,
        model_output_path=model_output_path,
        start_global_step=global_step,
        start_epoch=start_epoch,
        best_loss=best_loss,
    )


if __name__ == "__main__":
    main()
