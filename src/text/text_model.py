# src/text/text_model.py
import torch
from torch import nn
from transformers import AutoModel

class TextEmotionModel(nn.Module):
    def __init__(self, num_labels=14, model_name="hfl/chinese-roberta-wwm-ext-large"):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        # ✅ 支持 token_type_ids 或其他参数（不会报错）
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits
