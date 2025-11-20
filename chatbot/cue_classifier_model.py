"""
Cue Classifier Model
--------------------
A small MLP “Multi-Layer Perceptron.”
It is the simplest form of a neural network that sit on top of MiniLM embeddings.
This is a multi-label classifier:
    - each cue is an independent sigmoid output
    - loss = BCEWithLogitsLoss
"""

import torch
import torch.nn as nn


class CueClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        input_dim  = dimension of MiniLM embeddings (usually 384)
        num_labels = number of cue classes (11 in your dataset)
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor shaped [batch_size, input_dim]
        returns: logits [batch_size, num_labels]
        """
        return self.net(x)

    @staticmethod
    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Multi-label BCE loss (no softmax).
        `targets` = 0/1 for each cue label.
        """
        return nn.BCEWithLogitsLoss()(logits, targets)

    @staticmethod
    def predict_proba(logits: torch.Tensor) -> torch.Tensor:
        """
        Returns sigmoid probabilities in [0, 1].
        """
        return torch.sigmoid(logits)

    @staticmethod
    def predict_labels(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Convert logits to 0/1 predictions.
        """
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()
