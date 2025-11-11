import torch
from torch import nn
from transformers import Wav2Vec2Model


class Wav2Vec2ForMultiLabel(nn.Module):
    def __init__(self, num_labels=14, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_labels)

    def forward(self, input_values, attention_mask=None, **kwargs):
        """
        允许 **kwargs 接收多余参数（如 future processor 字段）
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits
