# arifllm

Sıfırdan Türkçe Büyük Dil Modeli (LLM) eğitmek için uçtan uca araçlar. Ham metin kaynaklarından tokenizer eğitimine, binary veri hazırlığından PyTorch tabanlı Transformer eğitimine ve Gradio ile metin üretimine kadar tüm adımları kapsar.

Hedef donanım: **NVIDIA RTX 4060 (8 GB VRAM)** üzerinde çalışabilen küçük–orta ölçekli modeller (6M–218M parametre).

## Pipeline

```
Ham Veri (PDF / EPUB / mC4 / TXT)
        ↓  araçlar/
    .txt dosyaları (≤256 MB parçalar)
        ↓  tokenleştirme/
kendi_tokenizerim.json (Türkçe BPE)
        ↓  tokenleştirme/
veriseti0001.bin, veriseti0002.bin … (uint16)
        ↓  eğitim/
model_rtx4060_*.pt
        ↓  arayüz_ui/
    Metin üretimi (Gradio)
```

## Klasör Yapısı

| Klasör | Açıklama |
|--------|----------|
| [`araçlar/`](araçlar/) | Ham veriyi `.txt` formatına dönüştürme araçları |
| [`tokenleştirme/`](tokenleştirme/) | Türkçe BPE tokenizer eğitimi ve binary tokenizasyon |
| [`eğitim/`](eğitim/) | LLM eğitim scripti ve RTX 4060 için hazır YAML konfigürasyonları |
| [`arayüz_ui/`](arayüz_ui/) | Eğitilmiş modeli test etmek için Gradio web arayüzü |

## Modüller

### 1. Veri Hazırlama (`araçlar/`)

| Script | Görev |
|--------|-------|
| `pdf2txt.py` | PDF → TXT (PyMuPDF veya pdfminer) |
| `epub2txt.py` | EPUB → TXT (toplu dönüştürme) |
| `mC42txt.py` | Hugging Face mC4 Türkçe verisini TXT'ye çevirir |
| `txt_böl.py` | 256 MB'dan büyük `.txt` dosyalarını parçalar |

```bash
python araçlar/pdf2txt.py H:\ham_veri H:\data\all_txt
python araçlar/mC42txt.py H:\mc4 H:\data\all_txt\mc4 --lang tr
python araçlar/txt_böl.py --input-dir H:\data\all_txt
```

Ek bağımlılıklar: `pymupdf` veya `pdfminer.six` (PDF), `ebooklib` + `beautifulsoup4` (EPUB).

### 2. Tokenleştirme (`tokenleştirme/`)

Türkçe için optimize edilmiş BPE tokenizer eğitir ve metinleri eğitime hazır binary dosyalara dönüştürür.

- **Algoritma:** BPE (Hugging Face `tokenizers`)
- **Normalizasyon:** Unicode NFKC + `casefold()` (Türkçe büyük/küçük harf)
- **Varsayılan vocab:** 65.535 — 300 GB+ veri için [131.072 önerilir](tokenleştirme/README.md)
- **Çıktı formatı:** `veriseti0001.bin` … (uint16, little-endian, max 256 MB/dosya)

```bash
cd tokenleştirme
pip install -r requirements.txt

python build_tokenizer.py --vocab-size 65535 --min-frequency 100 --lowercase \
    --input-dir "H:/data/all_txt" --output-dir .

python tokenize_to_binary.py --max-workers 8 \
    --input-dir "H:/data/all_txt" --output-dir "H:/data/data_v2" \
    --tokenizer-path "kendi_tokenizerim.json"
```

Detaylı rehber: [`tokenleştirme/README.md`](tokenleştirme/README.md)

### 3. Model Eğitimi (`eğitim/`)

PyTorch tabanlı `SimpleTransformer` modeli eğitir:

- Token + positional embedding, multi-head attention, weight tying
- Next-token prediction (CrossEntropyLoss), AdamW optimizer
- Chunk tabanlı veri yükleme (büyük veri setleri RAM'e sığmaz)
- Gradient accumulation, AMP / bfloat16 mixed precision
- Adım bazlı checkpoint ve `--resume` ile devam

```bash
cd eğitim
pip install -r requirements.txt

# Küçük model (~6M) — hızlı test
python train.py --config config-rtx4060_20m.yaml

# Orta model (~60M)
python train.py --config config-rtx4060_60m.yaml

# Büyük model (~218M) — 151 GB veri
python train.py --config config-rtx4060_200m.yaml

# Checkpoint'ten devam
python train.py --config config-rtx4060_200m.yaml \
    --resume H:/data/models/model_rtx4060_200m_ckpt_step_5000.pt
```

Detaylı rehber: [`eğitim/README.md`](eğitim/README.md)

#### Hazır Konfigürasyonlar

| Dosya | Parametre | Hedef veri | Peak VRAM (RTX 4060) |
|-------|-----------|------------|----------------------|
| `config-rtx4060_20m.yaml` | ~6M | 256 MB | ~2.1 GB |
| `config-rtx4060_60m.yaml` | ~60M | 256 MB | ~3.4 GB |
| `config-rtx4060_200m.yaml` | ~218M | 151 GB | ~5.5–6.5 GB |
| `config.example.yaml` | Genel | Özelleştirilebilir | — |

> **Not:** `config-rtx4060_20m.yaml` dosya adı "20m" olsa da ~6M parametre hedefler. `config-rtx4060_4M.yaml` geçerli bir config değildir; 4B model fizibilite notları içerir.

YAML dosyalarındaki `data_dir` ve `model_output_path` yollarını (`H:/data/...`) kendi ortamınıza göre düzenleyin.

### 4. Inference Arayüzü (`arayüz_ui/`)

Gradio tabanlı web arayüzü ile eğitilmiş modeli yükleyip metin üretimi test edin.

```bash
cd arayüz_ui
pip install gradio torch pyyaml tokenizers
python app.py
```

> **Bilinen sorun:** `app.py` şu an `train.train` modülünü import ediyor; gerçek klasör adı `eğitim`. Çalıştırmadan önce import yolunu düzeltmeniz gerekebilir.

## Sistem Gereksinimleri

| Gereksinim | Minimum | Önerilen |
|------------|---------|----------|
| Python | 3.8+ | 3.10+ |
| GPU | CUDA destekli (opsiyonel) | NVIDIA RTX 4060 8 GB |
| RAM | 8 GB | 32 GB |
| Disk | Proje + model | 300 GB+ (büyük Türkçe veri seti) |

## Bağımlılıklar

Merkezi `pyproject.toml` yok; her modülün kendi `requirements.txt` dosyası var:

| Modül | Temel paketler |
|-------|----------------|
| `eğitim/` | `torch`, `tokenizers`, `tqdm`, `numpy`, `pyyaml` |
| `tokenleştirme/` | `tokenizers` |
| `arayüz_ui/` | `gradio`, `torch`, `pyyaml`, `tokenizers` |

## Hızlı Başlangıç

```bash
git clone <repo-url> arifllm
cd arifllm

# 1. Ham veriyi TXT'ye çevir
python araçlar/pdf2txt.py ./ham_veri ./data/all_txt

# 2. Tokenizer eğit ve binary'ye dönüştür
cd tokenleştirme && pip install -r requirements.txt
python build_tokenizer.py --input-dir ../data/all_txt --output-dir .
python tokenize_to_binary.py --input-dir ../data/all_txt --output-dir ../data/data_v2
cd ..

# 3. Model eğit
cd eğitim && pip install -r requirements.txt
python train.py --config config-rtx4060_60m.yaml
cd ..

# 4. Metin üret (Gradio)
cd arayüz_ui && pip install gradio
python app.py
```

## Lisans

MIT
