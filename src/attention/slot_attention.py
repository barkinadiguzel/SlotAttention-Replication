import torch
import torch.nn as nn
import math
from .qkv_projection import QKVProjection
from .slot_update import SlotUpdate


class SlotAttention(nn.Module):
    def __init__(self, num_slots, input_dim, slot_dim, num_iters=3, eps=1e-8):
        super().__init__()

        self.num_slots = num_slots
        self.num_iters = num_iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.qkv = QKVProjection(input_dim, slot_dim)
        self.update = SlotUpdate(slot_dim)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

    def forward(self, inputs):
        B, N, D = inputs.shape

        inputs = self.norm_inputs(inputs)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = torch.exp(self.slots_sigma).expand(B, self.num_slots, -1)

        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.num_iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            q, k, v = self.qkv(inputs, slots_norm)

            attn_logits = torch.einsum("bnd,bkd->bnk", k, q) * self.scale
            attn = torch.softmax(attn_logits, dim=-1)

            attn = attn + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)

            updates = torch.einsum("bnk,bnd->bkd", attn, v)

            slots = self.update(slots_prev, updates)

        return slots, attn
