import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, out_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 5, padding=2)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2)
        self.conv4 = nn.Conv2d(hidden_dim, out_dim, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        B, D, H, W = x.shape
        x = x.view(B, D, H * W)
        x = x.permute(0, 2, 1)  

        return x
