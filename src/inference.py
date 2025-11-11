import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ✅ 自动设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'STHeiti', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 修复路径 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from transformers import BertTokenizer, Wav2Vec2Processor
from src.text.text_model import TextEmotionModel
from src.audio.audio_model import Wav2Vec2ForMultiLabel


LABELS = [
    "快乐", "愤怒", "悲伤", "惊讶", "厌恶",
    "恐惧", "平静", "紧张", "放松", "兴奋",
    "尴尬", "无聊", "困惑", "满意"
]

TEXT_MODEL_PATH = "models/text_emotion_model.pt"
AUDIO_MODEL_PATH = "models/audio_emotion_model.pt"


def predict_text(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = TextEmotionModel(num_labels=len(LABELS))
    if os.path.exists(TEXT_MODEL_PATH):
        model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs


def predict_audio(audio_path):
    from datasets import load_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForMultiLabel(num_labels=len(LABELS))
    if os.path.exists(AUDIO_MODEL_PATH):
        model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    ds = load_dataset("audiofolder", data_dir=os.path.dirname(audio_path))
    audio = ds["train"][0]["audio"]
    input_values = processor(audio["array"], sampling_rate=16000, return_tensors="pt", padding=True).input_values

    with torch.no_grad():
        logits = model(input_values.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs


def visualize_probabilities(probs, title="情感概率分布", save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    # 自动设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'STHeiti', 'PingFang SC']
    matplotlib.rcParams['axes.unicode_minus'] = False

    probs = np.array(probs)
    max_idx = np.argmax(probs)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(LABELS, probs, color=plt.cm.coolwarm(probs / probs.max()))

    # 高亮最高类别
    bars[max_idx].set_color("crimson")
    bars[max_idx].set_edgecolor("black")
    bars[max_idx].set_linewidth(2)

    # 在每个条上显示概率值
    for i, b in enumerate(bars):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{probs[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black" if i != max_idx else "crimson"
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.1)
    plt.ylabel("概率", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



def fuse_predictions(text_probs=None, audio_probs=None, text_weight=0.5, audio_weight=0.5):
    if text_probs is None:
        return audio_probs
    if audio_probs is None:
        return text_probs

    fused = text_weight * np.array(text_probs) + audio_weight * np.array(audio_probs)
    fused /= (text_weight + audio_weight)
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="输入文本（可选）")
    parser.add_argument("--audio", type=str, help="输入音频路径（可选）")
    parser.add_argument("--text_weight", type=float, default=0.5, help="文本模型权重")
    parser.add_argument("--audio_weight", type=float, default=0.5, help="音频模型权重")
    parser.add_argument("--save_plot", action="store_true", help="是否保存条形图")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    text_probs, audio_probs = None, None

    if args.text:
        print(f"📝 文本输入: {args.text}")
        text_probs = predict_text(args.text)
        visualize_probabilities(text_probs, title="文本模型输出", save_path="text_probs.png" if args.save_plot else None)

    if args.audio:
        print(f"🎧 音频输入: {args.audio}")
        audio_probs = predict_audio(args.audio)
        visualize_probabilities(audio_probs, title="音频模型输出", save_path="audio_probs.png" if args.save_plot else None)

    fused_probs = fuse_predictions(text_probs, audio_probs, args.text_weight, args.audio_weight)
    visualize_probabilities(fused_probs, title="融合后情感分布", save_path="fused_probs.png" if args.save_plot else None)

    top_idx = np.argmax(fused_probs)
    print(f"🌈 最终预测情感: {LABELS[top_idx]}  (概率: {fused_probs[top_idx]:.4f})")


if __name__ == "__main__":
    main()
