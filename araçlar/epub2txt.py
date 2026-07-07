import sys
import os
import glob
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def epub_to_txt(epub_path):
    """Tek bir epub dosyasını txt'ye çeviren mevcut fonksiyonunuz"""
    if not os.path.exists(epub_path):
        print(f"Hata: '{epub_path}' dosyası bulunamadı.")
        return

    base_name = os.path.splitext(epub_path)[0]
    txt_path = f"{base_name}.txt"

    print(f"'{os.path.basename(epub_path)}' dönüştürülüyor...")

    try:
        book = epub.read_epub(epub_path)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()
                    f.write(text + '\n')
        print(f"-> Başarılı! Kaydedildi: {os.path.basename(txt_path)}")
    except Exception as e:
        print(f"-> Hata: '{os.path.basename(epub_path)}' dönüştürülemedi: {e}")

def batch_convert(folder_path, mask):
    """Belirtilen klasördeki maskeye uyan tüm dosyaları bulup dönüştürür"""
    if not os.path.isdir(folder_path):
        print(f"Hata: '{folder_path}' geçerli bir klasör değil.")
        return

    # Klasör yolu ve maskeyi birleştiriyoruz (Örn: romanlar/*.epub)
    search_pattern = os.path.join(folder_path, mask)
    
    # Maskeye uyan tüm dosyaları listeliyoruz
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        print(f"'{folder_path}' içinde '{mask}' maskesine uygun dosya bulunamadı.")
        return

    print(f"Toplam {len(file_list)} dosya bulundu. İşlem başlıyor...\n" + "-"*40)
    
    # Tüm dosyaları sırayla fonksiyonumuza gönderiyoruz
    for file_path in file_list:
        epub_to_txt(file_path)
        
    print("-"*40 + "\nTüm dönüştürme işlemleri tamamlandı!")

if __name__ == "__main__":
    # Kullanıcı parametreleri eksik girdiyse rehber gösteriyoruz
    if len(sys.argv) < 3:
        print("Kullanım: python epub2txt_batch.py <klasor_yolu> <dosya_maskesi>")
        print("Örnek 1:  python epub2txt_batch.py ./romanlar *.epub")
        print("Örnek 2:  python epub2txt_batch.py C:/Kitaplar/ Macera*.epub")
    else:
        target_folder = sys.argv[1]
        file_mask = sys.argv[2]
        batch_convert(target_folder, file_mask)