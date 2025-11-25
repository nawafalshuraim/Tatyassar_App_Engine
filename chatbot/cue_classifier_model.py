"""
A small MLP “Multi-Layer Perceptron.”
It is the simplest form of a neural network that receive from MiniLM embeddings.
This is a multi-label classifier:
    - each cue is an independent sigmoid output
    - loss = BCEWithLogitsLoss
"""

import torch
import torch.nn as nn


class CueClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int, ## 384 from MiniLM embeddings
        num_labels: int, #11 cues
        hidden_dim: int = 256, #256 neurons inside
        dropout: float = 0.1, #randomly turns off 10% to prevent memorization
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
   
        return self.net(x)

    @staticmethod
    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
   
        # Since many cues can exist at the same time we use BCEWithLogitsLoss
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
        why 0.5 ? Balanced for now
        If probability ≥ 0.5 → predict cue present  
        If probability < 0.5 → predict cue not present
        Convert logits to 0/1 predictions.
        """
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()
