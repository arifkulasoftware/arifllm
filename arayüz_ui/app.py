import os
import sys
import torch
import torch.nn as nn
import yaml
import gradio as gr
from tokenizers import Tokenizer

# Projenin kök dizinini sys.path'e ekliyoruz
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from train.train import SimpleTransformer

# Cihazı belirliyoruz (GPU varsa kullanılır, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer varsayılan yolu
tokenizer_path = os.path.join(base_dir, "tokenleştirme", "kendi_tokenizerim.json")

# Global model ve tokenizer durumunu tutan değişkenler
current_model = None
tokenizer = None
current_context_length = 256

@torch.no_grad()
def generate(model, idx, max_new_tokens, context_length, temperature=0.7):
    """
    Modelden yeni kelimeler (tokenler) üretmek için otoregresif döngü.
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Context boyutu sınırını geçmemesi için son kısmı kırpıyoruz
        idx_cond = idx[:, -context_length:]
        # Modelden logits değerlerini alıyoruz
        logits = model(idx_cond)  # Şekil: (batch_size, vocab_size)
        
        # Temperature (yaratıcılık derecesi) kontrolü
        if temperature <= 0.0:
            # Greedy decoding (en yüksek olasılıklı olanı doğrudan al)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Logit değerlerini temperature'a bölüp olasılık dağılımı elde etme
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            # Olasılıklara göre rastgele örnekleme
            idx_next = torch.multinomial(probs, num_samples=1)
            
        # Yeni üretilen tokeni mevcut context'e ekliyoruz
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

def load_model(config_name_or_path, weights_path):
    """
    Belirtilen konfigürasyon ve ağırlık dosyasını kullanarak modeli yükler.
    """
    global current_model, tokenizer, current_context_length
    
    try:
        # 1. Tokenizer Yükleme
        if not os.path.exists(tokenizer_path):
            return f"❌ Tokenizer dosyası bulunamadı:\n{tokenizer_path}"
        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        
        # 2. Config Dosyası Yolu Belirleme
        config_path = config_name_or_path
        if config_name_or_path == "60M Model (Lokal)":
            config_path = os.path.join(base_dir, "train", "config-rtx4060_60m.yaml")
        elif config_name_or_path == "20M Model (H: Sürücüsü)":
            config_path = os.path.join(base_dir, "train", "config-rtx4060_20m.yaml")
            
        if not os.path.exists(config_path):
            return f"❌ Config dosyası bulunamadı:\n{config_path}"
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        model_cfg = config.get("model", {})
        current_context_length = model_cfg.get("context_length", 256)
        
        # 3. SimpleTransformer Modelini Oluşturma
        model = SimpleTransformer(
            vocab_size=vocab_size,
            embedding_dim=model_cfg.get("embedding_dim", 256),
            num_layers=model_cfg.get("num_layers", 4),
            num_heads=model_cfg.get("num_heads", 8),
            ff_dim=model_cfg.get("ff_dim", 1024),
            context_length=current_context_length,
            dropout=model_cfg.get("dropout", 0.1),
            tie_weights=model_cfg.get("tie_weights", True)
        )
        
        # 4. Model Ağırlıklarını Yükleme
        if not os.path.exists(weights_path):
            return f"❌ Model ağırlık dosyası bulunamadı:\n{weights_path}"
            
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        
        current_model = model
        
        total_params = sum(p.numel() for p in model.parameters())
        status_msg = (
            f"✅ Model başarıyla yüklendi!\n"
            f"- Cihaz: {device.type.upper()}\n"
            f"- Parametre Sayısı: {total_params:,}\n"
            f"- Bağlam Genişliği (Context): {current_context_length}\n"
            f"- Kelime Haznesi (Vocab): {vocab_size}"
        )
        return status_msg
        
    except Exception as e:
        current_model = None
        return f"❌ Model yüklenirken hata oluştu:\n{str(e)}"

def predict(prompt, creativity, max_tokens):
    """
    Gradio arayüzünden gelen girdileri alıp metin üretimi yapar.
    """
    if not prompt or not prompt.strip():
        return "Lütfen bir başlangıç metni yazın."
        
    if current_model is None:
        return "Hata: Yüklü bir model bulunmuyor. Lütfen Model Ayarları bölümünden modeli yükleyin."
        
    try:
        # Kullanıcının metnini token ID listesine çevir
        encoded = tokenizer.encode(prompt).ids
        if len(encoded) == 0:
            return "Geçersiz veya boş başlangıç cümlesi."
            
        context = torch.tensor([encoded], dtype=torch.long, device=device)
        
        # Otoregresif olarak token üret
        generated = generate(
            model=current_model,
            idx=context,
            max_new_tokens=int(max_tokens),
            context_length=current_context_length,
            temperature=float(creativity)
        )
        
        # ID listesini tekrar okunabilir metne çevir
        output_text = tokenizer.decode(generated[0].tolist())
        return output_text
    except Exception as e:
        return f"Metin üretilirken hata oluştu: {str(e)}"

def update_preset_paths(preset_name):
    """
    Şablon seçimine göre konfigürasyon ve ağırlık yollarını günceller.
    """
    if preset_name == "60M Model (Lokal)":
        cfg = os.path.join(base_dir, "train", "config-rtx4060_60m.yaml")
        wgt = os.path.join(base_dir, "train", "models", "model_rtx4060_60m.pt")
    elif preset_name == "20M Model (H: Sürücüsü)":
        cfg = os.path.join(base_dir, "train", "config-rtx4060_20m.yaml")
        wgt = "H:/data/model/model_rtx4060_20M.pt"
    else:  # Özel (Custom)
        cfg = ""
        wgt = ""
    return cfg, wgt

# Premium Arayüz Tasarımı ve CSS Kodları
custom_css = """
.header-card {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
    padding: 24px;
    border-radius: 12px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}
.header-card h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: white !important;
    margin: 0 0 8px 0 !important;
}
.header-card p {
    font-size: 1.1rem !important;
    color: #e0f2fe !important;
    margin: 0 !important;
}
.status-box {
    border-left: 4px solid #4f46e5 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"), css=custom_css) as demo:
    
    # HTML Banner
    gr.HTML(
        """
        <div class="header-card">
            <h1>🤖 Benim İlk Lokal LLM Asistanım</h1>
            <p>Sıfırdan eğittiğiniz transformer dil modelini test edin! Metin başlangıcı girerek devamını üretmesini sağlayın.</p>
        </div>
        """
    )
        
    with gr.Row():
        # Sol Sütun: Ayarlar ve Parametreler
        with gr.Column(scale=1):
            
            with gr.Accordion("⚙️ Model Yükleme Ayarları", open=True):
                preset_selector = gr.Dropdown(
                    choices=["60M Model (Lokal)", "20M Model (H: Sürücüsü)", "Özel (Custom)"],
                    value="60M Model (Lokal)",
                    label="Model Şablonu"
                )
                
                config_input = gr.Textbox(
                    value=os.path.join(base_dir, "train", "config-rtx4060_60m.yaml"),
                    label="Model Config (YAML) Yolu",
                    placeholder="Konfigürasyon dosyası yolunu girin..."
                )
                
                weights_input = gr.Textbox(
                    value=os.path.join(base_dir, "train", "models", "model_rtx4060_60m.pt"),
                    label="Model Ağırlıkları (.pt) Yolu",
                    placeholder="Model ağırlıkları dosya yolunu girin..."
                )
                
                load_btn = gr.Button("🔄 Model Yükle / Güncelle", variant="secondary")
                status_box = gr.Textbox(
                    label="Yükleme Durumu",
                    value="Bekleniyor... Modeli yüklemek için butona tıklayın.",
                    lines=4,
                    interactive=False,
                    elem_classes=["status-box"]
                )

            with gr.Group():
                slider = gr.Slider(
                    minimum=0.1, 
                    maximum=1.5, 
                    value=0.7, 
                    step=0.05, 
                    label="Yaratıcılık Derecesi (Temperature)",
                    info="Düşük değerler tutarlı, yüksek değerler yaratıcı çıktılar üretir."
                )
                max_tokens = gr.Slider(
                    minimum=10, 
                    maximum=500, 
                    value=150, 
                    step=10, 
                    label="Maksimum Token Sayısı"
                )

        # Sağ Sütun: Çıktı ve prompt alanı
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Başlangıç Cümlesi (Prompt)", 
                placeholder="Modelin devam ettirmesini istediğiniz bir cümle yazın...", 
                lines=5
            )
            submit_btn = gr.Button("✍️ Metin Üret", variant="primary")
            
            output_text = gr.Textbox(
                label="Modelin Yanıtı / Tamamlaması", 
                lines=12
            )

    # Tetikleyiciler (Event Handlers)
    preset_selector.change(
        fn=update_preset_paths,
        inputs=preset_selector,
        outputs=[config_input, weights_input]
    )
    
    load_btn.click(
        fn=load_model,
        inputs=[config_input, weights_input],
        outputs=status_box
    )
    
    submit_btn.click(
        fn=predict, 
        inputs=[input_text, slider, max_tokens], 
        outputs=output_text
    )

    # Sayfa ilk yüklendiğinde otomatik model yüklemeyi dene
    demo.load(
        fn=load_model,
        inputs=[config_input, weights_input],
        outputs=status_box
    )

if __name__ == "__main__":
    demo.launch()
