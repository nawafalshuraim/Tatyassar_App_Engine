"""
CueClassifier: Multi-Label Psychological Cue Detection
------------------------------------------------------
- Lightweight MLP for MiniLM embeddings (384-dim)
- BCEWithLogitsLoss for independent cue prediction
- Threshold is NOT fixed; tune externally per cue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CueClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        num_labels: int = 11,
        hidden_dim: int = 256,
        dropout: float = 0.15
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_labels = num_labels

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    # Forward Pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)

        # stability clamp:
        return logits.clamp(min=-50, max=50)

    # Loss
    @staticmethod
    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.BCEWithLogitsLoss()(logits, targets.float())

    # Probability Output
    @staticmethod
    def predict_proba(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    # Labels with dynamic threshold
    # (threshold passed externally)
    @staticmethod
    def predict_labels(
        logits: torch.Tensor, 
        threshold: float | list | torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # vector threshold support (per cue)
        if isinstance(threshold, (list, torch.Tensor)):
            threshold = torch.tensor(threshold, device=probs.device).float()
        else:
            threshold = float(threshold)

        return (probs >= threshold).float()

