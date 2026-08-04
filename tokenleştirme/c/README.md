# Pure C BPE Tokenizer Trainer

`build_tokenizer.py` ile aynı işi yapan, Hugging Face `tokenizers` kütüphanesi **kullanmayan** C uygulaması.

## Özellikler

- Rekürsif `.txt` dosya tarama
- NFKC normalizasyon (Windows: `NormalizeString`)
- `--lowercase` için casefold (Türkçe `İ` → `i` dahil, Windows üzerinde)
- Whitespace ön-tokenizasyon
- BPE eğitimi (`vocab_size`, `min_frequency`)
- Hugging Face uyumlu `kendi_tokenizerim.json` çıktısı
- Log ve checkpoint dosyası desteği
- `max-files`, `max-lines`, `max-bytes` limitleri

## Derleme (Windows / MSVC)

```bat
cd tokenleştirme\c
cl /O2 /W4 /std:c11 build_tokenizer.c utf.c bpe.c /Fe:build_tokenizer.exe Normaliz.lib
```

## Derleme (GCC / Clang)

```bash
cd tokenleştirme/c
make
```

Linux'ta NFKC için `libicu` normalizasyon kütüphanesi (`-lnormaliz`) gerekebilir.

## Kullanım

```bash
build_tokenizer.exe --input-dir "F:/data/all_txt" --output-dir "." --vocab-size 131072 --min-frequency 2 --lowercase --log-file tokenizer_training.log
```

Python sürümüyle aynı parametreler:

| Parametre | Açıklama |
|-----------|----------|
| `--input-dir` | Kaynak dizin (zorunlu) |
| `--output-dir` | Çıktı dizini (varsayılan: `.`) |
| `--vocab-size` | Hedef vocab boyutu (varsayılan: 32000) |
| `--min-frequency` | Minimum birleşim frekansı (varsayılan: 1000) |
| `--max-files` | Dosya limiti |
| `--max-lines` | Satır limiti |
| `--max-bytes` | Byte limiti |
| `--lowercase` | Casefold uygula |
| `--log-file` | Log dosyası |
| `--checkpoint-file` | Checkpoint dosyası |

Çıktı: `{output-dir}/kendi_tokenizerim.json`

## Dosya yapısı

```
c/
  build_tokenizer.c   # CLI, dosya okuma, ana akış
  utf.c / utf.h       # UTF-8, NFKC, casefold, whitespace bölme
  bpe.c / bpe.h       # BPE eğitimi ve JSON yazımı
  common.h            # Hash map ve yardımcı yapılar
  Makefile
```

## Notlar

- Büyük veri setlerinde bellek kullanımı Python sürümünden düşük olmalı; yine de milyonlarca benzersiz kelime için RAM gerekir.
- Linux/macOS'ta NFKC tam desteği için ICU kurulumu önerilir; ICU yoksa ham metin kullanılır.
- Üretilen JSON, `tokenize_to_binary.py` ile doğrudan kullanılabilir.
