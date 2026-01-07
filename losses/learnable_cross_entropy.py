import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableWeightedCrossEntropy(nn.Module):
    def __init__(self, n_classes, init_weights=None, eps=1e-8):
        super().__init__()

        if init_weights is None:
            init_weights = torch.ones(n_classes)

        # log-weights entrenables
        self.log_weights = nn.Parameter(torch.log(init_weights))
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: (B, C)
        targets: (B,)
        """
        log_probs = F.log_softmax(logits, dim=1)  # (B, C)

        # Seleccionar log-prob de la clase correcta
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # pesos por muestra
        weights = torch.exp(self.log_weights)              # (C,)
        sample_weights = weights[targets]                  # (B,)

        loss = (sample_weights * nll).mean()
        return loss


