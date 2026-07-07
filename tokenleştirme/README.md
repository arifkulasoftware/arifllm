# Tokenizer builder

This script scans a directory recursively for `.txt` files and trains a BPE tokenizer optimized for Turkish. The resulting tokenizer JSON is saved as `kendi_tokenizerim.json`.

Usage:

```bash
python build_tokenizer.py --input-dir "F:/My App/ai/data/all_txt/books" --output-dir . --vocab-size 65535
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- The script applies Unicode NFKC normalization and `casefold()` to the training lines to improve Turkish casing handling.
- Adjust `--vocab-size` and `--min-frequency` as needed for dataset size.
