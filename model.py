"""Fully connected neural network for MNIST digit classification."""

import torch
from torch import nn


class DigitClassifier(nn.Module):
    """Map a 28x28 grayscale image to 10 class scores (logits).

    Architecture (forward pass):
        input  -> (batch_size, 1, 28, 28)
        flatten -> (batch_size, 784)
        Linear(784, 128) + ReLU -> (batch_size, 128)
        Linear(128, 64) + ReLU  -> (batch_size, 64)
        Linear(64, 10)          -> (batch_size, 10)  # raw logits, no softmax
    """

    def __init__(self):
        super().__init__()
        # Flatten turns each image into a vector of 784 pixel values.
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(in_features=28 * 28, out_features=128)
        self.hidden2 = nn.Linear(in_features=128, out_features=64)
        self.output = nn.Linear(in_features=64, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (batch_size, 1, 28, 28)
        flattened = self.flatten(images)
        hidden1_activated = self.relu(self.hidden1(flattened))
        hidden2_activated = self.relu(self.hidden2(hidden1_activated))
        logits = self.output(hidden2_activated)
        # logits: (batch_size, 10). CrossEntropyLoss applies log-softmax internally.
        return logits
