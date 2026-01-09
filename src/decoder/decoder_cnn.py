import torch.nn as nn
import torch.nn.functional as F


class DecoderCNN(nn.Module):
    def __init__(self, slot_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(slot_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 4, 1)  
        )

    def forward(self, x):
        return self.net(x)
