# src/train_audio.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import json
from sklearn.metrics import f1_score, accuracy_score

NUM_LABELS = 14
SAMPLE_RATE = 16000  # Wav2Vec2 默认采样率
THRESHOLD = 0.5       # 多标签预测阈值

class AudioDataset(Dataset):
    def __init__(self, data_path):
        """
        data_path: JSONL 文件，每行 {"audio_path": "...", "labels": [0,1,...]}
        """
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        waveform, sr = torchaudio.load(item['audio_path'])

        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

        waveform = waveform.mean(dim=0)  # 合并为单声道
        inputs = self.processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.squeeze(0)

        labels = torch.tensor(item['labels'], dtype=torch.float)
        return input_values, labels

class Wav2Vec2ForMultiLabel(nn.Module):
    def __init__(self, num_labels=NUM_LABELS):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_values):
        attention_mask = (input_values != 0).long()
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits

def collate_fn(batch):
    input_values = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch]).float()

    input_lengths = [x.shape[0] for x in input_values]
    max_len = max(input_lengths)
    padded_inputs = torch.zeros(len(batch), max_len)
    for i, x in enumerate(input_values):
        padded_inputs[i, :x.shape[0]] = x
    return padded_inputs, labels

def train_audio_model(data_path='data/audio_data.jsonl', epochs=3, lr=2e-5, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AudioDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = Wav2Vec2ForMultiLabel()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fct = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        all_labels = []
        all_preds = []
        running_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_values, labels = batch
            input_values = input_values.to(device)
            labels = labels.to(device)

            logits = model(input_values)
            loss = loss_fct(logits, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 计算预测
            preds = torch.sigmoid(logits).detach().cpu() >= THRESHOLD
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu())

        # 统计指标
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        f1 = f1_score(all_labels, all_preds, average='micro')
        acc = accuracy_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}: Loss={running_loss/len(dataloader):.4f}, F1-micro={f1:.4f}, Accuracy={acc:.4f}")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/audio_emotion_model.pt')
    print('✅ Multi-label audio model saved to models/audio_emotion_model.pt')

if __name__ == '__main__':
    train_audio_model()
