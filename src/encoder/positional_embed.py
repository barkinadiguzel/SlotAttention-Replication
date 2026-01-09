import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(2, dim)

    def forward(self, x, H, W):
        device = x.device
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )

        coords = torch.stack([x_coords, y_coords], dim=-1)  
        coords = coords.view(H * W, 2)
        pos = self.linear(coords)  

        return x + pos.unsqueeze(0)
