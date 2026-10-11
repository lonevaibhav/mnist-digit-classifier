"""Evaluate a saved DigitClassifier on the MNIST test set and plot training loss."""

import json
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import DigitClassifier


def _allow_torchvision_import() -> None:
    """Work around Windows policies that block NumPy's random DLL."""
    try:
        np.random  # noqa: B018
    except Exception:
        placeholder = types.ModuleType("numpy.random")
        sys.modules["numpy.random"] = placeholder
        np.random = placeholder  # type: ignore[attr-defined]


_allow_torchvision_import()
from torchvision import datasets, transforms  # noqa: E402

DATA_DIR = Path("data")
MODEL_PATH = Path("models") / "mnist_fc.pth"
HISTORY_PATH = Path("artifacts") / "training_history.json"
CONFUSION_PATH = Path("artifacts") / "confusion_matrix.png"
LOSS_PLOT_PATH = Path("artifacts") / "training_loss.png"

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_model(device: torch.device) -> DigitClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained weights at {MODEL_PATH}. Run `python train.py` first."
        )
    model = DigitClassifier()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # inference mode: disables dropout / uses running batch-norm stats
    return model


def create_test_loader(batch_size: int = 256) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate(model: DigitClassifier, dataloader: DataLoader, device: torch.device):
    correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    # no_grad skips storing tensors needed to compute gradients — faster and uses less memory.
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    test_accuracy = 100.0 * correct / total_samples
    return test_accuracy, all_labels, all_predictions


def plot_confusion_matrix(labels, predictions) -> None:
    matrix = torch.zeros((10, 10), dtype=torch.int64)
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted] += 1

    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xlabel("Predicted digit")
    axis.set_ylabel("Actual digit")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    axis.set_title("MNIST test set confusion matrix")
    for row in range(10):
        for col in range(10):
            count = int(matrix[row, col].item())
            axis.text(col, row, str(count), ha="center", va="center", fontsize=8)
    CONFUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150)
    try:
        plt.show()
    except Exception:
        pass
    plt.close()
    print(f"Saved confusion matrix to {CONFUSION_PATH}")


def plot_training_loss() -> None:
    if not HISTORY_PATH.exists():
        print(f"No training history found at {HISTORY_PATH}; skip loss plot.")
        return

    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    losses = history.get("train_loss", [])
    accuracies = history.get("train_accuracy", [])
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss")
    plt.title("Training loss across epochs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    try:
        plt.show()
    except Exception:
        pass
    plt.close()

    print(f"Saved training-loss plot to {LOSS_PLOT_PATH}")
    if accuracies:
        print("Training accuracy per epoch (from train.py, not the test set):")
        for epoch, accuracy in enumerate(accuracies, start=1):
            print(f"  epoch {epoch}: {accuracy:.2f}%")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    model = load_trained_model(device)
    test_loader = create_test_loader()
    test_accuracy, labels, predictions = evaluate(model, test_loader, device)

    print()
    print("Accuracy comparison")
    print("  Training accuracy: how well the model fit the 60,000 training images.")
    print("  Test accuracy:     how well it generalizes to 10,000 unseen images.")
    print(f"  Test accuracy: {test_accuracy:.2f}%")
    print("  Re-read train.py output for the matching training accuracy of this run.")

    plot_confusion_matrix(labels, predictions)
    plot_training_loss()


if __name__ == "__main__":
    main()
