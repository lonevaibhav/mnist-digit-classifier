"""Load saved weights and predict the digit in one MNIST test image."""

import argparse
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

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
PREDICTION_PATH = Path("artifacts") / "prediction.png"

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_model(device: torch.device) -> DigitClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained weights at {MODEL_PATH}. Run `python train.py` first, "
            "then call this script without retraining."
        )
    model = DigitClassifier()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_test_sample(index: int):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    # Unnormalized copy is easier to display as a handwritten digit.
    display_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    model_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )
    if index < 0 or index >= len(model_dataset):
        raise IndexError(f"index must be between 0 and {len(model_dataset) - 1}.")

    display_image, actual_label = display_dataset[index]
    model_image, _ = model_dataset[index]
    return display_image, model_image, actual_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one MNIST test digit with saved weights.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the test image (0-9999).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model = load_trained_model(device)
    display_image, model_image, actual_label = load_test_sample(args.index)

    # Add a batch dimension: (1, 28, 28) -> (1, 1, 28, 28)
    batch = model_image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)  # shape: (1, 10)
        probabilities = torch.softmax(logits, dim=1)
        predicted_digit = int(logits.argmax(dim=1).item())
        confidence = float(probabilities[0, predicted_digit].item())

    print(f"Test image index: {args.index}")
    print(f"Predicted digit:  {predicted_digit}  (confidence {confidence:.2%})")
    print(f"Actual label:     {actual_label}")

    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.imshow(display_image.squeeze(0), cmap="gray")
    plt.title(f"Predicted: {predicted_digit}  |  Actual: {actual_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PREDICTION_PATH, dpi=150)
    try:
        plt.show()
    except Exception:
        pass
    plt.close()
    print(f"Saved preview to {PREDICTION_PATH}")


if __name__ == "__main__":
    main()
