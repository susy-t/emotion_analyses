# src/fusion_model.py
import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, text_dim=14, audio_dim=14, hidden_dim=64, num_labels=14):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid()
        )

    def forward(self, text_vec, audio_vec):
        fused = torch.cat((text_vec, audio_vec), dim=1)
        return self.fc(fused)
