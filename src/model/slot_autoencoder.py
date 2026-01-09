import torch.nn as nn
from src.encoder.cnn_encoder import CNNEncoder
from src.encoder.positional_embed import PositionalEmbedding
from src.attention.slot_attention import SlotAttention
from src.decoder.decoder_cnn import DecoderCNN
from src.decoder.compositor import Compositor


class SlotAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = CNNEncoder()
        self.pos_embed = PositionalEmbedding(config.slot_dim)
        self.slot_attention = SlotAttention(
            config.num_slots,
            config.encoder_dim,
            config.slot_dim,
            config.num_iters
        )

        self.decoder = DecoderCNN(config.slot_dim)
        self.compositor = Compositor()

    def forward(self, x):
        B, C, H, W = x.shape

        feats = self.encoder(x)
        feats = self.pos_embed(feats, H, W)

        slots, attn = self.slot_attention(feats)

        slots = slots.unsqueeze(-1).unsqueeze(-1)
        slots = slots.expand(-1, -1, -1, H, W)

        slots = slots.reshape(B * slots.shape[1], slots.shape[2], H, W)
        decoded = self.decoder(slots)
        decoded = decoded.view(B, -1, 4, H, W)

        recon, masks = self.compositor(decoded)

        return recon, masks, slots
