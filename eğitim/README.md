# LLM Model Eğitim (train.py)

Bu script, tokenize edilmiş binary dosyalardan LLM (Large Language Model) modeli eğitir.

## Özellikler

- ✅ `kendi_tokenizerim.json` tokenizer dosyasını yükler
- ✅ Binary format token dosyaları okur (uint16 format)
- ✅ Transformer tabanlı LLM modeli eğitir
- ✅ PyTorch CUDA desteği
- ✅ Parametrik konfigürasyon
- ✅ Checkpoint ve model kayıt desteği
- ✅ Gradient clipping ve weight decay

## Kurulum

1. **Gerekli paketleri yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

## Kullanım

### 1. Config Dosyası ile (Önerilen) ⭐

Config dosyasında tüm parametreleri tanımlayın:

```bash
# Dosyanızı kopyalayın
cp config.example.yaml config.yaml

# Dosyayı ihtiyacınıza göre düzenleyin
# (Aşağıdaki "Konfigürasyon Dosyası" bölümüne bakın)

# Sonra çalıştırın
python train.py --config config.yaml
```

### 2. Config + Komut Satırı Override

Config dosyası bazlı parametreleri komut satırından override edin:

```bash
python train.py --config config.yaml --epochs 5 --batch-size 16
```

### 3. Sadece Komut Satırı (Eski Yöntem)

```bash
python train.py \
    --tokenizer-path ../tokenleştirme/kendi_tokenizerim.json \
    --data-dir ./data \
    --data-pattern "veriseti*.bin" \
    --model-output-path ./models/model.pt \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 0.0001
```

## Parametreler

### Zorunlu Parametreler:
- `--tokenizer-path`: Tokenizer JSON dosyasının yolu (kendi_tokenizerim.json)
- `--data-dir`: Binary dosyaların bulunduğu dizin
- `--model-output-path`: Eğitilen modelin kaydedileceği yol

**Veri Parametreleri:**
- `--data-pattern`: Binary dosyaları bulmak için glob pattern (default: `veriseti*.bin`)

**Model Parametreleri:**
- `--embedding-dim`: Embedding boyutu (default: 256)
- `--num-layers`: Transformer katman sayısı (default: 4)
- `--num-heads`: Attention head sayısı (default: 8)
- `--ff-dim`: Feed-forward katman boyutu (default: 1024)
- `--context-length`: Context (sequence) uzunluğu (default: 512)
- `--dropout`: Dropout oranı (default: 0.1)

**Eğitim Parametreleri:**
- `--epochs`: Eğitim epoch sayısı (default: 10)
- `--batch-size`: Batch boyutu (default: 32)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--weight-decay`: Weight decay (L2 regularization) (default: 0.01)

**Diğer Parametreler:**
- `--device`: Eğitim cihazı (`cuda` veya `cpu`, default: cuda varsa cuda)
- `--save-checkpoint-every`: Kaç epoch sonrası checkpoint kaydet (default: 0 = devre dışı)

## Konfigürasyon Dosyası (YAML Format)

**Önerilen Kullanım:** Tüm parametreleri `config.yaml` dosyasında tanımlayın.

### Örnek Konfigürasyonlar:

#### 📋 Genel Konfigürasyon (config.example.yaml):
```yaml
# Tokenizer ve veri dizinleri
tokenizer_path: ../tokenleştirme/kendi_tokenizerim.json
data_dir: ./data

# Veri parametreleri
data:
  pattern: "veriseti*.bin"

# Model mimarisi
model:
  embedding_dim: 512
  num_layers: 6
  num_heads: 8
  ff_dim: 2048
  context_length: 512
  dropout: 0.1

# Eğitim parametreleri
training:
  epochs: 20
  batch_size: 64
  learning_rate: 0.0001
  weight_decay: 0.01
  device: cuda

# Checkpoint ve model kayıt
checkpoint:
  save_every: 5
  model_output_path: ./models/model.pt
```

#### 🎮 RTX 4060 8GB İçin Optimize (config-rtx4060.yaml):
**6 Milyon Parametre - 256MB Veri Seti için:**
```yaml
tokenizer_path: ../tokenleştirme/kendi_tokenizerim.json
data_dir: ./data

data:
  pattern: "veriseti*.bin"

model:
  embedding_dim: 256      # 6M param hedefi
  num_layers: 5           # 6M param hedefi
  num_heads: 8
  ff_dim: 1024            # Bellek tasarrufu
  context_length: 256     # Bellek tasarrufu
  dropout: 0.2

training:
  epochs: 8
  batch_size: 16          # RTX 4060 8GB için optimal
  learning_rate: 0.0001
  weight_decay: 0.01
  device: cuda

checkpoint:
  save_every: 2
  model_output_path: ./models/model_rtx4060.pt
```

**RTX 4060 Özellikleri:**
- VRAM: 8GB
- Peak memory usage: ~2.1GB (safe!)
- Estimated per-epoch time: ~17 dakika
- Total training time (8 epochs): ~2.3 saat

### Başlamak için:
```bash
# RTX 4060 için:
cp config-rtx4060.yaml config.yaml

# veya direkt RTX 4060 config'i kullan:
python train.py --config config-rtx4060.yaml

# Batch size'ı CLI'den override et:
python train.py --config config-rtx4060.yaml --batch-size 8 --epochs 15
```

### Config + CLI Override:
Config dosyasındaki parametreleri komut satırından override edebilirsiniz:
```bash
# config.yaml'daki epochs 20'si, bu komutla 5 olacak
python train.py --config config.yaml --epochs 5 --batch-size 128
```

## Veri Formatı

Binary dosyalar aşağıdaki formatta olmalıdır:
- Dosya adı: `veriseti0001.bin`, `veriseti0002.bin` vb.
- İçerik: Sırasıyla yazılan uint16 (2 byte) token ID'leri
- Sıra: little-endian (`<H`)

Bu format, `tokenize_to_binary.py` scripti tarafından otomatik olarak oluşturulur.

## Model Mimarisi

- **Embedding Layer**: Token embedding + positional encoding
- **Transformer Encoder**: Multi-head attention ile N katman
- **Output Head**: Vocab boyutunda projection layer
- **Loss Function**: Cross Entropy (next-token prediction)
- **Optimizer**: AdamW (adaptive learning rate)

## Çıkış

Eğitim tamamlandığında:
- Ana model: `model_output_path` konumunda kaydedilir
- Checkpoints: `model_output_path` dizininde `checkpoint_epoch_X.pt` formatında kaydedilir
- En iyi model otomatik olarak kaydedilir

## Model Yükleme

Eğitilen modeli kullanmak için:
```python
import torch
from train import SimpleTransformer

# Model oluştur
model = SimpleTransformer(vocab_size=50000, ...)
# Kaydedilen state dictionary'yi yükle
model.load_state_dict(torch.load('./models/model.pt'))
model.eval()
```

## Sistem Gereksinimleri

- Python 3.8+
- NVIDIA GPU (CUDA desteği için önerilen, CPU'da da çalışır)
- Minimum 8GB RAM
- Veri dosyaları için yeterli disk alanı

## Performans İpuçları

- GPU kullanılabilirse `--device cuda` kullanın (10-50x daha hızlı)
- `--batch-size` belleğe göre ayarlayın (daha büyük = daha hızlı)
- `--context-length` azaltırsanız bellek kullanımı azalır
- `--embedding-dim` ve `--num-layers` azaltmak modeli hızlandırır
- `--save-checkpoint-every` ayarlayın uzun eğitimler için

## Sorun Giderme

**CUDA Out of Memory:**
- Batch size'ı azaltın: `--batch-size 16`
- Context length'i azaltın: `--context-length 256`
- Model boyutunu azaltın: `--embedding-dim 128 --num-layers 2`

**Eğitim çok yavaş:**
- GPU kullandığınızdan emin olun: `--device cuda`
- Batch size'ı artırın: `--batch-size 128`

**NaN loss değerleri:**
- Learning rate'i düşürün: `--learning-rate 0.00001`
- Weight decay'i ayarlayın: `--weight-decay 0.1`

## Lisans

MIT
