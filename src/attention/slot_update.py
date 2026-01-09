import torch
import torch.nn as nn


class SlotUpdate(nn.Module):
    def __init__(self, slot_dim, hidden_dim=128):
        super().__init__()

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.norm = nn.LayerNorm(slot_dim)

    def forward(self, slots, updates):
        B, K, D = slots.shape

        slots = slots.view(B * K, D)
        updates = updates.view(B * K, D)

        slots = self.gru(updates, slots)
        slots = slots.view(B, K, D)

        slots = slots + self.mlp(self.norm(slots))

        return slots
