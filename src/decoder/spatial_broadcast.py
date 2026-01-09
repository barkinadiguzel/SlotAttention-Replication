import torch


class SpatialBroadcast:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, slots):
        B, K, D = slots.shape
        H, W = self.resolution

        slots = slots.view(B, K, 1, 1, D)
        slots = slots.expand(B, K, H, W, D)

        return slots
