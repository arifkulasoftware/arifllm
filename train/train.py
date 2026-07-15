#!/usr/bin/env python3
"""
LLM Model Training Script

Bu script:
- kendi_tokenizerim.json dosyasını yükler
- Binary dosyalarından (veriseti*.bin) eğitim verilerini okur
- PyTorch kullanarak LLM modeli eğitir
- Eğitilen modeli kaydeder

Kullanım (Config dosyası ile - önerilen):
    cp config.example.yaml config.yaml
    # config.yaml'ı düzenleyin
    python train.py --config config.yaml

Kullanım (Config + CLI override):
    python train.py --config config.yaml --epochs 5 --batch-size 16

Kullanım (Sadece komut satırı):
    python train.py \
        --tokenizer-path ../tokenleştirme/kendi_tokenizerim.json \
        --data-dir ./data \
        --model-output-path ./models/model.pt \
        --epochs 10 \
        --batch-size 32 \
        --resume path/to/checkpoint.pt

"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from tokenizers import Tokenizer
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None


class BinaryTokenDataset(IterableDataset):
    """
    Binary token dosyalarından token ID'leri okur.
    Her veri noktası context_length kadar token içerir.
    """

    def __init__(
        self,
        data_dir: Path,
        data_pattern: str,
        context_length: int = 512,
        vocab_size: int = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_pattern = data_pattern
        self.context_length = context_length
        self.vocab_size = vocab_size

        # Binary dosyaları bul
        self.bin_files = sorted(self.data_dir.glob(data_pattern))
        if not self.bin_files:
            raise FileNotFoundError(
                f"Binary dosya bulunamadı: {self.data_dir}/{data_pattern}"
            )
        print(f"Bulunan {len(self.bin_files)} binary dosya")

    def load_tokens_from_file(self, file_path: Path):
        """Binary dosyadan token ID'lerini yükle"""
        try:
            import numpy as np
            return np.fromfile(file_path, dtype=np.uint16)
        except Exception as e:
            print(f"Hata: {file_path} okunurken: {e}")
            import numpy as np
            return np.array([], dtype=np.uint16)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tüm token'ları belleğe yükle ve context_length boyutunda
        öğrenme örnekleri oluştur.
        """
        import numpy as np
        all_tokens_list = []

        # Tüm binary dosyaları yükle
        for bin_file in self.bin_files:
            print(f"Yükleniyor: {bin_file.name}")
            tokens = self.load_tokens_from_file(bin_file)
            if len(tokens) > 0:
                all_tokens_list.append(tokens)

        if not all_tokens_list:
            print("Uyarı: Token bulunamadı!")
            return

        all_tokens = np.concatenate(all_tokens_list)
        print(f"Toplam {len(all_tokens):,} token yüklendi")

        # context_length boyutunda öğrenme örnekleri oluştur
        num_examples = len(all_tokens) // (self.context_length + 1)
        print(f"Üretilebilir örnek sayısı: {num_examples:,}")

        for i in range(num_examples):
            start_idx = i * self.context_length
            end_idx = start_idx + self.context_length

            # Giriş: context_length kadar token
            input_tokens = all_tokens[start_idx:end_idx]
            # Hedef: bir sonraki token (next-token prediction)
            target_token = all_tokens[end_idx]

            # Eğer vocab_size belirtildiyse, kontrolü yap
            if self.vocab_size is not None:
                if (input_tokens >= self.vocab_size).any():
                    continue
                if target_token >= self.vocab_size:
                    continue

            yield (
                torch.tensor(input_tokens, dtype=torch.long),
                torch.tensor(target_token, dtype=torch.long),
            )


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

        # Token embedding
        token_emb = self.token_embedding(x)  # (batch, seq_len, embedding_dim)

        # Positional encoding
        positions = torch.arange(
            seq_length, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        pos_emb = self.positional_embedding(positions)  # (1, seq_len, embedding_dim)

        # Embedding'leri birleştir
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)

        # Transformer encoder
        transformer_output = self.transformer_encoder(embeddings)

        # Son token'ın çıkışını al
        last_token_output = transformer_output[:, -1, :]

        # Çıkış projeksiyonu
        logits = self.output_head(last_token_output)  # (batch, vocab_size)

        return logits


def get_vocab_size(tokenizer_path: Path) -> int:
    """Tokenizer'dan vocab boyutunu al"""
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer.get_vocab_size()


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    YAML config dosyasını yükle
    
    Args:
        config_path: Config dosyasının yolu
        
    Returns:
        Config sözlüğü
    """
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
        raise ValueError(f"Invalid config file: {config_path}")
    
    return config


def merge_config_and_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Config dosyasından parametreleri yükle ve merge et.
    Komut satırı parametreleri config dosyasını override eder.
    
    Args:
        config: Config sözlüğü
        args: Komut satırı argümanları
        
    Returns:
        Merge edilmiş argümanlar
    """
    if "tokenizer_path" in config:
        if not hasattr(args, "tokenizer_path") or args.tokenizer_path is None:
            args.tokenizer_path = config["tokenizer_path"]
    
    if "data_dir" in config:
        if not hasattr(args, "data_dir") or args.data_dir is None:
            args.data_dir = config["data_dir"]
    
    # Data parametreleri
    if "data" in config:
        data_config = config["data"]
        if "pattern" in data_config:
            if not hasattr(args, "data_pattern") or args.data_pattern is None:
                args.data_pattern = data_config["pattern"]
    
    # Model parametreleri
    if "model" in config:
        model_config = config["model"]
        if "embedding_dim" in model_config:
            if not hasattr(args, "embedding_dim") or args.embedding_dim is None:
                args.embedding_dim = model_config["embedding_dim"]
        if "num_layers" in model_config:
            if not hasattr(args, "num_layers") or args.num_layers is None:
                args.num_layers = model_config["num_layers"]
        if "num_heads" in model_config:
            if not hasattr(args, "num_heads") or args.num_heads is None:
                args.num_heads = model_config["num_heads"]
        if "ff_dim" in model_config:
            if not hasattr(args, "ff_dim") or args.ff_dim is None:
                args.ff_dim = model_config["ff_dim"]
        if "context_length" in model_config:
            if not hasattr(args, "context_length") or args.context_length is None:
                args.context_length = model_config["context_length"]
        if "dropout" in model_config:
            if not hasattr(args, "dropout") or args.dropout is None:
                args.dropout = model_config["dropout"]
        if "tie_weights" in model_config:
            if not hasattr(args, "tie_weights") or args.tie_weights is None:
                args.tie_weights = model_config["tie_weights"]
    
    # Eğitim parametreleri
    if "training" in config:
        training_config = config["training"]
        if "epochs" in training_config:
            if not hasattr(args, "epochs") or args.epochs is None:
                args.epochs = training_config["epochs"]
        if "batch_size" in training_config:
            if not hasattr(args, "batch_size") or args.batch_size is None:
                args.batch_size = training_config["batch_size"]
        if "learning_rate" in training_config:
            if not hasattr(args, "learning_rate") or args.learning_rate is None:
                args.learning_rate = training_config["learning_rate"]
        if "weight_decay" in training_config:
            if not hasattr(args, "weight_decay") or args.weight_decay is None:
                args.weight_decay = training_config["weight_decay"]
        if "device" in training_config:
            if not hasattr(args, "device") or args.device is None:
                args.device = training_config["device"]
    
    # Checkpoint parametreleri
    if "checkpoint" in config:
        checkpoint_config = config["checkpoint"]
        if "save_every" in checkpoint_config:
            if not hasattr(args, "save_checkpoint_every") or args.save_checkpoint_every is None:
                args.save_checkpoint_every = checkpoint_config["save_every"]
        if "model_output_path" in checkpoint_config:
            if not hasattr(args, "model_output_path") or args.model_output_path is None:
                args.model_output_path = checkpoint_config["model_output_path"]
        if "resume" in checkpoint_config:
            if not hasattr(args, "resume") or args.resume is None:
                args.resume = checkpoint_config["resume"]
    
    return args


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """Bir epoch eğitim yap ve ortalama loss'u döndür"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    with tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch") as pbar:
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits, target_ids)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="LLM Model Eğitim Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:

  1. Config dosyası ile (önerilen):
    python train.py --config config.yaml

  2. Config dosyası ve komut satırı override'ı:
    python train.py --config config.yaml --epochs 5 --batch-size 16

  3. Sadece komut satırı parametreleri:
    python train.py \\
      --tokenizer-path ../tokenleştirme/kendi_tokenizerim.json \\
      --data-dir ./data \\
      --model-output-path ./models/model.pt \\
      --epochs 10 \\
      --batch-size 32
        """,
    )

    # Config dosyası parametresi
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config dosyasının yolu (config parametreleri komut satırı ile override edilebilir)",
    )

    # Zorunlu parametreler (config olmadığında gerekli)
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer JSON dosyasının yolu (kendi_tokenizerim.json)",
    )
    parser.add_argument(
        "--data-dir", 
        default=None,
        help="Binary dosyaların bulunduğu dizin"
    )
    parser.add_argument(
        "--data-pattern",
        default=None,
        help="Binary dosyaları bulmak için glob pattern (default: veriseti*.bin)",
    )
    parser.add_argument(
        "--model-output-path",
        default=None,
        help="Eğitilen modelin kaydedileceği yol",
    )

    # Model parametreleri
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding boyutu (default: 256)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=None, help="Transformer katman sayısı (default: 4)"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Attention head sayısı (default: 8)",
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=None,
        help="Feed-forward katman boyutu (default: 1024)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context uzunluğu (default: 512)",
    )
    parser.add_argument(
        "--dropout", type=float, default=None, help="Dropout oranı (default: 0.1)"
    )

    # Eğitim parametreleri
    parser.add_argument(
        "--epochs", type=int, default=None, help="Eğitim epoch sayısı (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch boyutu (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (default: 0.01)",
    )

    # Diğer parametreler
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Eğitim cihazı (default: cuda varsa cuda, yoksa cpu)",
    )
    parser.add_argument(
        "--save-checkpoint-every",
        type=int,
        default=None,
        help="Her N epoch sonrası checkpoint kaydet (default: 0)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Eğitime devam etmek için yüklenecek checkpoint (.pt dosyası) yolu (default: None)",
    )
    parser.add_argument(
        "--tie-weights",
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        default=None,
        help="Embedding ve output head ağırlıklarını paylaş (Weight Tying) (default: True)",
    )

    args = parser.parse_args()

    # Config dosyasını yükle
    if args.config:
        print(f"Config dosyası yükleniyor: {args.config}")
        config = load_config(args.config)
        args = merge_config_and_args(config, args)
    
    # Varsayılan değerleri ayarla (None ise)
    if args.tokenizer_path is None:
        parser.error("--tokenizer-path parametresi gerekli (config veya komut satırında)")
    if args.data_dir is None:
        parser.error("--data-dir parametresi gerekli (config veya komut satırında)")
    if args.model_output_path is None:
        parser.error("--model-output-path parametresi gerekli (config veya komut satırında)")
    
    # Varsayılan değerleri ayarla
    if args.data_pattern is None:
        args.data_pattern = "veriseti*.bin"
    if args.embedding_dim is None:
        args.embedding_dim = 256
    if args.num_layers is None:
        args.num_layers = 4
    if args.num_heads is None:
        args.num_heads = 8
    if args.ff_dim is None:
        args.ff_dim = 1024
    if args.context_length is None:
        args.context_length = 512
    if args.dropout is None:
        args.dropout = 0.1
    if args.epochs is None:
        args.epochs = 10
    if args.batch_size is None:
        args.batch_size = 32
    if args.learning_rate is None:
        args.learning_rate = 0.0001
    if args.weight_decay is None:
        args.weight_decay = 0.01
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.save_checkpoint_every is None:
        args.save_checkpoint_every = 0
    if args.tie_weights is None:
        args.tie_weights = True

    # Parametreleri doğrula
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    model_output_path = Path(args.model_output_path).expanduser().resolve()

    if not tokenizer_path.exists():
        print(f"Hata: Tokenizer dosyası bulunamadı: {tokenizer_path}")
        sys.exit(1)

    if not data_dir.exists():
        print(f"Hata: Veri dizini bulunamadı: {data_dir}")
        sys.exit(1)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cihazı ayarla
    device = torch.device(args.device)
    print(f"Kullanılan cihaz: {device}")

    # Vocab boyutunu al
    print(f"Tokenizer yükleniyor: {tokenizer_path}")
    vocab_size = get_vocab_size(tokenizer_path)
    print(f"Vocab boyutu: {vocab_size}")

    # Dataset oluştur
    print(f"Dataset oluşturuluyor: {data_dir}/{args.data_pattern}")
    dataset = BinaryTokenDataset(
        data_dir=data_dir,
        data_pattern=args.data_pattern,
        context_length=args.context_length,
        vocab_size=vocab_size,
    )

    # DataLoader oluştur
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Model oluştur
    print(f"Model oluşturuluyor...")
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

    # Model parametrelerini yazdır
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametreler: {total_params:,}")
    print(f"Eğitilebilir parametreler: {trainable_params:,}")

    # Optimizer ve loss function
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Checkpoint'ten yükleme (Resume)
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if resume_path.exists():
            print(f"Checkpoint yükleniyor: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
            if "loss" in checkpoint:
                best_loss = checkpoint["loss"]
            print(f"✓ Checkpoint başarıyla yüklendi. Eğitim epoch {start_epoch} konumundan devam edecek. (Önceki Loss: {best_loss:.4f})")
        else:
            print(f"Hata: Belirtilen checkpoint dosyası bulunamadı: {resume_path}")
            sys.exit(1)

    # Eğitim döngüsü
    print(f"\nEğitim başlıyor...")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
    print("-" * 80)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            avg_loss = train_epoch(
                model, dataloader, optimizer, criterion, device, epoch
            )
            print(f"Epoch {epoch}/{args.epochs} - Ortalama Loss: {avg_loss:.4f}")

            # Checkpoint kaydet
            if args.save_checkpoint_every > 0 and epoch % args.save_checkpoint_every == 0:
                checkpoint_path = (
                    model_output_path.parent
                    / f"checkpoint_epoch_{epoch}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint kaydedildi: {checkpoint_path}")

            # En iyi modeli sakla
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), model_output_path)
                print(f"✓ Model kaydedildi: {model_output_path}")

    except KeyboardInterrupt:
        print("\nEğitim kullanıcı tarafından durduruldu.")
        # Son modeli kaydet
        torch.save(model.state_dict(), model_output_path)
        print(f"Model kaydedildi: {model_output_path}")

    print("-" * 80)
    print(f"Eğitim tamamlandı!")
    print(f"Son model kaydedildi: {model_output_path}")
    print(f"En iyi loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
