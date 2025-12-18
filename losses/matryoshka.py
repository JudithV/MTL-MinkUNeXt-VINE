# Miguel Hernández University of Elche
# Institute for Engineering Research of Elche (I3E)
# Automation, Robotics and Computer Vision lab (ARCV)
# Author: Judith Vilella Cantos

import torch
import torch.nn as nn
import torch.nn.functional as F

class MatryoshkaLoss(nn.Module):
    def __init__(self, base_loss_fn, dims, weights=None, normalize=True):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.dims = dims
        self.normalize = normalize

        if weights is None:
            self.weights = [1.0] * len(dims)
        else:
            assert len(weights) == len(dims)
            self.weights = weights

    def forward(self, embeddings, positives_mask, negatives_mask):
        total_loss = 0.0
        all_stats = {}

        for d, w in zip(self.dims, self.weights):
            emb_d = embeddings[:, :d]

            if self.normalize:
                emb_d = F.normalize(emb_d, dim=1)

            loss_d, stats_d = self.base_loss_fn(
                emb_d, positives_mask, negatives_mask
            )

            total_loss = total_loss + w * loss_d
            if d == self.dims[-1]:
                for k, v in stats_d.items():
                    all_stats[k] = v


        all_stats['loss'] = total_loss.detach()

        return total_loss, all_stats

