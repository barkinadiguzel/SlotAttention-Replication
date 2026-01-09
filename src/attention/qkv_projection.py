import torch.nn as nn


class QKVProjection(nn.Module):
    def __init__(self, input_dim, slot_dim):
        super().__init__()

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)

    def forward(self, inputs, slots):
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        q = self.to_q(slots)

        return q, k, v
