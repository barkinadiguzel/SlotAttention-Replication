import torch
import torch.nn.functional as F


class Compositor:
    def __call__(self, x):
        rgb = x[:, :, :3]
        masks = x[:, :, 3:4]

        masks = torch.softmax(masks, dim=1)
        recon = torch.sum(rgb * masks, dim=1)

        return recon, masks
