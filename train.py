"""Download MNIST, train DigitClassifier, and save weights plus training history."""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import DigitClassifier


def _allow_torchvision_import() -> None:
    """Work around Windows policies that block NumPy's random DLL.

    torchvision imports numpy.random during package setup. Training itself
    does not need that DLL, so a placeholder is enough when the real module
    cannot load.
    """
    try:
        np.random  # noqa: B018 - trigger lazy import
    except Exception:
        placeholder = types.ModuleType("numpy.random")
        sys.modules["numpy.random"] = placeholder
        np.random = placeholder  # type: ignore[attr-defined]


_allow_torchvision_import()
from torchvision import datasets, transforms  # noqa: E402

DATA_DIR = Path("data")
MODEL_PATH = Path("models") / "mnist_fc.pth"
HISTORY_PATH = Path("artifacts") / "training_history.json"

# MNIST mean/std from the training set. Normalization helps Adam converge faster.
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_dataloader(train: bool, batch_size: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # PIL image [0, 255] -> float tensor [0, 1], shape (1, 28, 28)
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=train,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def explain_sample_shape(images: torch.Tensor, labels: torch.Tensor) -> None:
    print("Sample batch shapes")
    print(f"  images: {tuple(images.shape)}  # (batch_size, channels, height, width)")
    print(f"  labels: {tuple(labels.shape)}  # (batch_size,) integer class ids 0-9")
    print(
        "  After Flatten in the model, each image becomes a vector of "
        f"{images.shape[1] * images.shape[2] * images.shape[3]} features."
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()  # enable dropout/batch-norm training behavior (none here, but good habit)
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # forget gradients from the previous batch
        logits = model(images)  # forward pass
        loss = loss_fn(logits, labels)
        loss.backward()  # fill .grad on every parameter
        optimizer.step()  # Adam updates weights using those gradients

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return average_loss, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MNIST fully connected classifier.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of passes over the training set.")
    parser.add_argument("--batch-size", type=int, default=64, help="Images per training batch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    train_loader = create_dataloader(train=True, batch_size=args.batch_size)
    sample_images, sample_labels = next(iter(train_loader))
    explain_sample_shape(sample_images, sample_labels)

    model = DigitClassifier().to(device)
    # CrossEntropyLoss = log-softmax + negative log likelihood. Do not softmax in forward().
    loss_fn = nn.CrossEntropyLoss()
    # Adam adapts the step size per parameter; a solid default for this network.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {"train_loss": [], "train_accuracy": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train loss: {train_loss:.4f}  "
            f"train accuracy: {train_accuracy:.2f}%"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved weights to {MODEL_PATH}")
    print(f"Saved training history to {HISTORY_PATH}")
    print("Next: run python evaluate.py to measure test accuracy (not the same as train accuracy).")


if __name__ == "__main__":
    main()
