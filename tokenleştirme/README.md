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


Genel Notlar
txt dosya veri kaynağı 300Gb böyle bir eğitim seti kaynağı için türkçe eğitimde ideal vocab_size ne olmalı?
10:43 AM
Türkçe için ideal vocab_size hesaplaması birkaç faktöre bağlıdır. Bunları sırayla inceleyelim:

Türkçenin Dilbilimsel Yapısı (Neden Kritik?)
Türkçe, sondan eklemeli (agglutinative) bir dildir. Tek bir kelime köküne onlarca ek gelebilir:

ev → evlendirilemeyeceklerinden (8 ek, 1 kök)

Bu durum, İngilizce'ye kıyasla çok daha fazla benzersiz kelime formu üretir. Yani vocab_size doğrudan Türkçe'nin morfolojik zenginliğini karşılayabilecek büyüklükte olmalıdır.

Vocab Size Seçimini Belirleyen 3 Faktör
1. BPE Subword Granülaritesi
vocab_size	Token Başına Ortalama Karakter	Türkçe Kelime Karşılama	Sorun
8.000 – 16.000	2–3 karakter	Çok düşük	Kelimeler onlarca parçaya bölünür, bağlam çok kısa kalır
32.000	3–5 karakter	Orta	İngilizce için iyi, Türkçe için yetersiz
50.000 – 65.535	4–7 karakter	İyi	Mevcut tokenizer'ınız bu aralıkta
100.000 – 131.072	6–9 karakter	Çok İyi	Türkçe için önerilen aralık
200.000+	8–12 karakter	Mükemmel	Model embedding katmanı çok büyür, verimlilik düşer
2. 300 GB txt Veri Seti ile Unique Token Doyumu
BPE tokenizer eğitiminde veri büyüdükçe benzersiz subword sayısı artar ama bir noktada doyuma (saturation) ulaşır:

Veri Boyutu	Türkçe BPE Doyum Noktası (Tahmini)
1–10 GB	~50.000 – 80.000 token
10–50 GB	~80.000 – 120.000 token
50–300 GB	~100.000 – 150.000 token
300 GB+	~130.000 – 200.000 (azalan getiri)
300 GB veri için 65.535 token yetersiz kalacaktır — özellikle Türkçe'nin morfolojik çeşitliliği göz önüne alındığında.

3. Model Boyutu ile Embedding Katmanı Maliyeti
vocab_size doğrudan embedding matrisini büyütür ve VRAM tüketir:

vocab_size	embedding_dim=512	embedding_dim=1024	embedding_dim=2048
65.535	134 MB	268 MB	537 MB
100.000	205 MB	409 MB	819 MB
131.072	268 MB	537 MB	1.07 GB
200.000	410 MB	819 MB	1.64 GB
RTX 4060 (8 GB) ile 200M parametreli model hedefleniyorsa 131.072 vocab_size VRAM açısından hâlâ uygulanabilir.

Türkçe için Bilimsel Referanslar
Model	Dil	Vocab Size	Veri Boyutu	Sonuç
BERTurk	Türkçe	128.000	~35 GB	Türkçe subword için yeterli
mT5	Çok dilli	250.000	Petabayt	Geniş dil kapsamı
TurkishBERTweet	Türkçe (sosyal medya)	52.000	~5 GB	Yeterli (küçük veri)
LLaMA-3	İngilizce ağırlıklı	128.000	15T token	Modern standart
Mistral-7B	İngilizce	32.000	—	İngilizce için yeterli, Türkçe için az
Kesin Tavsiye
300 GB Türkçe veri seti için ideal vocab_size: 131.072

Neden 131.072?

$2^{17} = 131.072$ → Hesaplamada ikinin kuvveti olduğu için GPU tensor operasyonlarında otomatik hizalama (alignment) avantajı sağlar.
300 GB'lık Türkçe metindeki morfolojik çeşitliliği karşılamaya yetecek granülaritede.
RTX 4060 üzerinde 120M–200M parametreli modelde embedding katmanı ~268–537 MB arası VRAM kullanır — bu kabul edilebilir bir değer.
Mevcut 65.535 tokenizer'ınızı yeniden eğitmeniz gerekecektir.
IMPORTANT

kendi_tokenizerim.json dosyanızı 65.535 vocab_size ile eğittiniz. 300 GB veri ve 131.072 hedefiyle tokenizer'ı yeniden eğitmeniz (build_tokenizer.py --vocab-size 131072) ve ardından tüm binary veri setini yeniden tokenize etmeniz gerekecektir. Bu işlem tek seferlik bir maliyettir.