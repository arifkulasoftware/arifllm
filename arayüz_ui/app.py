import os
import sys
from pathlib import Path

import gradio as gr
import torch
import yaml
from tokenizers import Tokenizer

base_dir = Path(__file__).resolve().parent.parent
egitim_dir = base_dir / "eğitim"

if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from eğitim.train import SimpleTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_model = None
tokenizer = None
current_context_length = 256


def discover_config_files() -> list[str]:
    """eğitim/ altındaki geçerli config*.yaml dosyalarını listele."""
    configs: list[str] = []
    for config_path in sorted(egitim_dir.glob("config*.yaml")):
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict) and data.get("model"):
                configs.append(config_path.name)
        except Exception:
            continue
    return configs


def resolve_config_path(config_name: str) -> Path:
    return (egitim_dir / config_name).resolve()


def resolve_path(path_value: str, base_dir_for_relative: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir_for_relative / path).resolve()


def read_config(config_name: str) -> dict:
    config_path = resolve_config_path(config_name)
    if not config_path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict) or not config.get("model"):
        raise ValueError(f"Geçersiz config dosyası (model bölümü yok): {config_path}")

    return config


def get_paths_from_config(config_name: str) -> tuple[Path, Path, Path]:
    """Config'ten tokenizer, model ağırlık ve config dosya yollarını çıkar."""
    config_path = resolve_config_path(config_name)
    config = read_config(config_name)

    tokenizer_rel = config.get("tokenizer_path", "../tokenleştirme/kendi_tokenizerim.json")
    tokenizer_file = resolve_path(tokenizer_rel, config_path.parent)

    checkpoint_cfg = config.get("checkpoint", {})
    weights_rel = checkpoint_cfg.get("model_output_path")
    if not weights_rel:
        raise ValueError(
            f"Config dosyasında checkpoint.model_output_path tanımlı değil: {config_name}"
        )
    weights_file = resolve_path(weights_rel, config_path.parent)

    return config_path, tokenizer_file, weights_file


def describe_config(config_name: str) -> str:
    """Seçilen config için özet bilgi döndür."""
    try:
        config_path, tokenizer_file, weights_file = get_paths_from_config(config_name)
        model_cfg = read_config(config_name).get("model", {})
        lines = [
            f"Config: {config_path}",
            f"Tokenizer: {tokenizer_file}",
            f"Model ağırlıkları: {weights_file}",
            f"Mimari: embed={model_cfg.get('embedding_dim')} | "
            f"layers={model_cfg.get('num_layers')} | "
            f"heads={model_cfg.get('num_heads')} | "
            f"context={model_cfg.get('context_length')}",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"Config okunamadı: {exc}"


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_length, temperature=0.7):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        if temperature <= 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def load_model(config_name: str):
    """Seçilen config dosyasındaki parametrelere göre modeli yükle."""
    global current_model, tokenizer, current_context_length

    if not config_name:
        return "Lütfen bir config dosyası seçin."

    try:
        config_path, tokenizer_file, weights_file = get_paths_from_config(config_name)
        config = read_config(config_name)
        model_cfg = config["model"]

        if not tokenizer_file.exists():
            return f"❌ Tokenizer dosyası bulunamadı:\n{tokenizer_file}"

        if not weights_file.exists():
            return f"❌ Model ağırlık dosyası bulunamadı:\n{weights_file}"

        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        vocab_size = tokenizer.get_vocab_size()
        current_context_length = model_cfg.get("context_length", 256)

        model = SimpleTransformer(
            vocab_size=vocab_size,
            embedding_dim=model_cfg.get("embedding_dim", 256),
            num_layers=model_cfg.get("num_layers", 4),
            num_heads=model_cfg.get("num_heads", 8),
            ff_dim=model_cfg.get("ff_dim", 1024),
            context_length=current_context_length,
            dropout=model_cfg.get("dropout", 0.1),
            tie_weights=model_cfg.get("tie_weights", True),
        )

        checkpoint = torch.load(weights_file, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        current_model = model

        total_params = sum(p.numel() for p in model.parameters())
        return (
            f"✅ Model başarıyla yüklendi!\n"
            f"- Config: {config_path.name}\n"
            f"- Cihaz: {device.type.upper()}\n"
            f"- Parametre sayısı: {total_params:,}\n"
            f"- Context length: {current_context_length}\n"
            f"- Vocab size: {vocab_size:,}\n"
            f"- Ağırlıklar: {weights_file}"
        )

    except Exception as exc:
        current_model = None
        return f"❌ Model yüklenirken hata oluştu:\n{exc}"


def predict(prompt, creativity, max_tokens):
    if not prompt or not prompt.strip():
        return "Lütfen bir başlangıç metni yazın."

    if current_model is None:
        return "Hata: Yüklü model yok. Lütfen bir config seçip modeli yükleyin."

    try:
        encoded = tokenizer.encode(prompt).ids
        if len(encoded) == 0:
            return "Geçersiz veya boş başlangıç cümlesi."

        context = torch.tensor([encoded], dtype=torch.long, device=device)
        generated = generate(
            model=current_model,
            idx=context,
            max_new_tokens=int(max_tokens),
            context_length=current_context_length,
            temperature=float(creativity),
        )
        return tokenizer.decode(generated[0].tolist())
    except Exception as exc:
        return f"Metin üretilirken hata oluştu: {exc}"


CONFIG_CHOICES = discover_config_files()
DEFAULT_CONFIG = CONFIG_CHOICES[0] if CONFIG_CHOICES else None

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

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    css=custom_css,
) as demo:
    gr.HTML(
        """
        <div class="header-card">
            <h1>🤖 Benim İlk Lokal LLM Asistanım</h1>
            <p>Eğitim config dosyasını seçin; model mimarisi, tokenizer ve ağırlık yolu otomatik yüklensin.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Model Yükleme", open=True):
                config_selector = gr.Dropdown(
                    choices=CONFIG_CHOICES,
                    value=DEFAULT_CONFIG,
                    label="Eğitim Config Dosyası",
                    info="eğitim/ altındaki config*.yaml dosyaları",
                )

                config_info = gr.Textbox(
                    label="Config Özeti",
                    value=describe_config(DEFAULT_CONFIG) if DEFAULT_CONFIG else "Config bulunamadı.",
                    lines=6,
                    interactive=False,
                )

                load_btn = gr.Button("🔄 Model Yükle / Güncelle", variant="secondary")
                status_box = gr.Textbox(
                    label="Yükleme Durumu",
                    value="Bekleniyor... Bir config seçip modeli yükleyin.",
                    lines=6,
                    interactive=False,
                    elem_classes=["status-box"],
                )

            with gr.Group():
                slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.05,
                    label="Yaratıcılık Derecesi (Temperature)",
                    info="Düşük değerler tutarlı, yüksek değerler yaratıcı çıktılar üretir.",
                )
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=150,
                    step=10,
                    label="Maksimum Token Sayısı",
                )

        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Başlangıç Cümlesi (Prompt)",
                placeholder="Modelin devam ettirmesini istediğiniz bir cümle yazın...",
                lines=5,
            )
            submit_btn = gr.Button("✍️ Metin Üret", variant="primary")
            output_text = gr.Textbox(label="Modelin Yanıtı / Tamamlaması", lines=12)

    config_selector.change(
        fn=describe_config,
        inputs=config_selector,
        outputs=config_info,
    )

    load_btn.click(
        fn=load_model,
        inputs=config_selector,
        outputs=status_box,
    )

    submit_btn.click(
        fn=predict,
        inputs=[input_text, slider, max_tokens],
        outputs=output_text,
    )

    if DEFAULT_CONFIG:
        demo.load(
            fn=load_model,
            inputs=config_selector,
            outputs=status_box,
        )

if __name__ == "__main__":
    demo.launch()
