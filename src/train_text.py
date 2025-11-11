# src/train_text.py
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm

MODEL_NAME = 'hfl/chinese-roberta-wwm-ext-large'
NUM_LABELS = 14


def train_text_model(data_path='data/text_data.jsonl', epochs=3, lr=2e-5, batch_size=8):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)

    # Custom multi-label classification head
    class MultiLabelClassifier(torch.nn.Module):
        def __init__(self, base_model, num_labels):
            super().__init__()
            self.base_model = base_model
            hidden_size = base_model.config.hidden_size
            self.classifier = torch.nn.Linear(hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            logits = self.classifier(pooled_output)
            return logits

    model = MultiLabelClassifier(base_model, NUM_LABELS)
    model.to(device)

    # Load dataset
    dataset = load_dataset('json', data_files=data_path)['train']

    # Tokenize text
    def encode(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    dataset = dataset.map(encode, batched=True)

    # Convert labels to float tensors
    def convert_labels(example):
        example['labels'] = torch.tensor(example['labels'], dtype=torch.float)
        return example

    dataset = dataset.map(convert_labels)

    # Set dataset format
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Collate function to ensure correct batch shapes and float labels
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]).float()
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fct = torch.nn.BCEWithLogitsLoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = loss_fct(logits, batch['labels'])

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Ensure the 'models' directory exists and save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/text_emotion_model.pt')
    print('✅ Multi-label text model saved to models/text_emotion_model.pt')


if __name__ == '__main__':
    train_text_model()
