# Handwritten Digit Classifier (PyTorch)

A beginner-friendly MNIST project that trains a small fully connected network to recognize digits **0–9**. The code is split so you can see each PyTorch idea in isolation: tensors and data loading, `nn.Module`, the training loop, evaluation, and inference from saved weights.

## Project structure

```
mnist-classifier/
├── train.py           # download MNIST, train, save weights + loss history
├── model.py           # DigitClassifier (nn.Module)
├── evaluate.py        # test accuracy, confusion matrix, training-loss plot
├── predict.py         # load saved weights and classify one test image
├── requirements.txt
├── README.md
└── .gitignore
```

After you train, these folders appear locally (they are gitignored):

```
data/                  # torchvision MNIST download
models/mnist_fc.pth    # trained parameters (state_dict)
artifacts/             # plots and training_history.json
```

## Overall workflow

1. **`train.py`** downloads MNIST, converts images to tensors, trains `DigitClassifier` with Adam and `CrossEntropyLoss`, then writes `models/mnist_fc.pth`.
2. **`evaluate.py`** loads those weights, runs the **test** set with `model.eval()` and `torch.no_grad()`, prints test accuracy, and plots a confusion matrix plus the training-loss curve.
3. **`predict.py`** loads the same weights again (no retraining) and classifies a single test image.

Training accuracy answers “how well did we fit the 60,000 training images?” Test accuracy answers “how well do we generalize to 10,000 unseen images?” Those two numbers are not interchangeable.

---

## File-by-file walkthrough

### `model.py` — the network

**Purpose:** Define a class that PyTorch can train: a subclass of `nn.Module` with layers as attributes and a `forward()` method.

**Important pieces:**

| Piece | Role |
| --- | --- |
| `nn.Module` | Base class. Registers parameters so they appear in `model.parameters()` and move with `.to(device)`. |
| `nn.Flatten()` | `(batch, 1, 28, 28)` → `(batch, 784)`. |
| `nn.Linear(in, out)` | `y = xWᵀ + b`. These `W` and `b` tensors are the learnable weights. |
| `nn.ReLU()` | `max(0, x)` so the network can learn non-linear decision boundaries. |
| `forward()` | The computation graph used during the forward pass. |

**Tensor shapes:**

```
(batch, 1, 28, 28)
        → flatten → (batch, 784)
        → Linear 784→128 + ReLU → (batch, 128)
        → Linear 128→64  + ReLU → (batch, 64)
        → Linear 64→10          → (batch, 10)  logits
```

There is **no softmax in `forward()`**. `nn.CrossEntropyLoss` already applies log-softmax internally. Adding softmax yourself would distort the loss.

During training, `loss.backward()` writes a `.grad` tensor next to every parameter. `optimizer.step()` then changes those parameters. After a successful step, weights are slightly different; after `zero_grad()`, the stored gradients are cleared so the next batch does not accumulate the previous batch’s gradients (Adam here uses one batch at a time, not gradient accumulation).

### `train.py` — data, training loop, saving

**Purpose:** Load MNIST, print a sample tensor shape, run the explicit training loop, save `state_dict` and per-epoch history.

**Important imports / functions:**

| Name | Role |
| --- | --- |
| `transforms.ToTensor()` | PIL `[0, 255]` image → `float32` tensor in `[0, 1]` with shape `(C, H, W)`. |
| `transforms.Normalize(mean, std)` | `(x - mean) / std` using MNIST’s known mean `0.1307` and std `0.3081`. |
| `datasets.MNIST` | Downloads and caches the dataset under `data/`. |
| `DataLoader` | Yields batches. `shuffle=True` on train so each epoch sees a new order. |
| `nn.CrossEntropyLoss` | Standard loss for single-label classification with integer class ids. |
| `torch.optim.Adam` | Adaptive per-parameter learning rates; a reliable default vs. plain SGD. |
| `model.train()` | Training mode (dropout / batch-norm behave as in training). |

**Why this loss?** Each image has exactly one correct digit. Cross-entropy penalizes the model when the logit for the true class is not large compared with the other nine. Labels are integers `0–9`, which is the format `CrossEntropyLoss` expects (not one-hot vectors).

**Why Adam?** You would otherwise pick a single learning rate for every weight. Adam keeps running estimates of gradient mean and variance and scales each update. For this small fully connected net it usually converges quickly without extra tuning.

**What happens each batch:**

1. `images` / `labels` move to CPU or CUDA.
2. `optimizer.zero_grad()` clears old `.grad` values.
3. `logits = model(images)` — forward propagation.
4. `loss = loss_fn(logits, labels)` — a scalar tensor with `grad_fn`.
5. `loss.backward()` — reverse-mode autodiff fills `.grad`.
6. `optimizer.step()` — Adam updates `W` and `b`.

Average training loss and **training** accuracy are printed after every epoch.

### `evaluate.py` — test set and plots

**Purpose:** Measure generalization. This script never updates weights.

**`model.eval()` vs `model.train()`:**

- `train()`: layers like Dropout randomly zero activations; BatchNorm uses batch statistics.
- `eval()`: Dropout is turned off; BatchNorm uses stored running statistics.

This network has neither Dropout nor BatchNorm, but calling `eval()` is still the correct inference habit. If you later add Dropout and forget `eval()`, test accuracy will look noisier and usually worse.

**`torch.no_grad()`:** Autograd would otherwise keep intermediate tensors in case you call `backward()`. During evaluation you do not need gradients, so `no_grad()` saves memory and time and makes it obvious that weights must not change.

Test accuracy is computed on the 10,000-image **test** split. The confusion matrix is a 10×10 count of `(actual digit, predicted digit)` pairs on the test set. The loss plot is **training** loss from `train.py` — it is not test loss.

### `predict.py` — one-image inference

**Purpose:** Load `mnist_fc.pth` and classify one test example. If the file is missing, the script raises a clear error instead of silently training.

Softmax is applied **only here**, to turn logits into a confidence value for display. The predicted class is still `argmax` of the logits (same as argmax of softmax).

---

## Setup and installation

From a terminal, in this folder:

**Windows (PowerShell)**

```powershell
cd c:\Users\hp\OneDrive\Desktop\mnist-classifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd mnist-classifier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Google Colab**

Upload the `.py` files. Colab already includes PyTorch, so install only what is missing:

```python
!pip install matplotlib
```

Do **not** blindly `pip install -r requirements.txt` in Colab: that pin is a local CPU wheel (`torch==2.8.0`) and can replace Colab’s GPU build.

PyTorch will use CUDA when a GPU is available (`torch.cuda.is_available()`), otherwise CPU. The code does not require a GPU.

`requirements.txt` pins matching `torch` / `torchvision` versions. Mixing unrelated versions (for example a new torch with an old torchvision) often fails at import time.

If `Activate.ps1` is blocked on Windows, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Commands

Train (default: 10 epochs, batch size 64):

```bash
python train.py
```

Optional flags:

```bash
python train.py --epochs 8 --batch-size 128 --learning-rate 0.001
```

Evaluate on the test set:

```bash
python evaluate.py
```

Predict one test image (index `0` by default):

```bash
python predict.py
python predict.py --index 42
```

Plots are saved under `artifacts/` even if a window does not open (for example in Colab, call the same scripts or `%run`).

---

## Complete training pipeline

1. **Data:** 60,000 train / 10,000 test grayscale 28×28 digits.
2. **Tensor conversion:** `ToTensor` → shape `(1, 28, 28)`, values in `[0, 1]`.
3. **Normalization:** zero-center roughly and scale, which keeps activations in a range Adam handles well.
4. **Minibatch:** e.g. 64 images, shape `(64, 1, 28, 28)`, labels `(64,)`.
5. **Forward:** pixels → 128 ReLU units → 64 ReLU units → 10 logits.
6. **Loss:** Cross-entropy between logits and integer labels.
7. **Backward:** PyTorch builds a graph during the forward pass, then walks it backward to compute ∂loss/∂parameter.
8. **Update:** Adam changes every parameter a small amount to reduce the loss.
9. **Repeat** for all batches, then for each epoch.
10. **Save:** only the parameter tensors (`state_dict`), not Python code. `evaluate.py` / `predict.py` reconstruct the architecture from `model.py` and load those tensors.

---

## Expected output

Numbers depend on random initialization, hardware, and epoch count.

A **verified 2-epoch CPU run** on this project produced:

```
Epoch 1/2  train loss: 0.2757  train accuracy: 91.97%
Epoch 2/2  train loss: 0.1138  train accuracy: 96.53%
Test accuracy: 97.00%
Predicted digit:  7  (confidence 99.95%)
Actual label:     7
```

(The first MNIST test image is a 7.) Default `python train.py` uses **5** epochs, so your numbers will differ.

The block below is **illustrative** of typical console formatting, not a guarantee:

**`python train.py` (illustrative)**

```
Using device: cpu
Sample batch shapes
  images: (64, 1, 28, 28)  # (batch_size, channels, height, width)
  labels: (64,)  # (batch_size,) integer class ids 0-9
  After Flatten in the model, each image becomes a vector of 784 features.
Epoch 1/5  train loss: 0.3100  train accuracy: 91.20%
Epoch 2/5  train loss: 0.1300  train accuracy: 96.10%
...
Saved weights to models\mnist_fc.pth
```

A small fully connected MNIST model often reaches **high 90s percent** training accuracy in a few epochs. Test accuracy is usually a bit lower than the last training accuracy.

**`python evaluate.py` (illustrative)**

```
Test accuracy: 97.xx%
Saved confusion matrix to artifacts\confusion_matrix.png
```

Most mistakes, if any, often sit between similar shapes (for example 4 vs 9, or 3 vs 5).

**`python predict.py --index 0` (illustrative)**

```
Test image index: 0
Predicted digit:  7  (confidence 99.xx%)
Actual label:     7
```

The first MNIST test image is commonly a 7, but always trust the printed actual label.

---

## Common beginner errors

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `FileNotFoundError` for `mnist_fc.pth` | Evaluate/predict before training | Run `python train.py` first. |
| `RuntimeError: size mismatch` when loading | Architecture changed but old weights kept | Delete `models/mnist_fc.pth` and retrain. |
| Loss is `nan` | Learning rate far too large | Keep `--learning-rate 0.001` for Adam. |
| Accuracy stuck near 10% | Model not in train loop, or labels/logits swapped | Confirm `loss_fn(logits, labels)` and that `optimizer.step()` runs. |
| `softmax` + `CrossEntropyLoss` | Double log-softmax | Leave softmax out of `forward()`. |
| Train accuracy high, test much lower | Overfitting (more likely with extra-large nets) | Stop earlier, add dropout, or keep this small architecture. |
| `plt.show()` hangs in some terminals | No GUI backend | Plots are already saved under `artifacts/`. |
| `ImportError: NP_SUPPORTED_MODULES` or NumPy `DLL load failed` | Some Windows Application Control policies block `numpy.random` | The scripts already register a placeholder if that import fails. On a normal setup the real NumPy module is used. |
| `torch` and `torchvision` import crash | Unmatched package versions | Install the pair in `requirements.txt` (this repo was verified with torch 2.8.0 and torchvision 0.23.0). |
| CUDA out of memory | Batch too large on a small GPU | Lower `--batch-size` (for example 32). |

---

## Exercises

1. **Deeper net:** Add a third hidden layer of 32 units with ReLU. Retrain. Did test accuracy rise, fall, or stay similar?
2. **SGD vs Adam:** Switch `torch.optim.Adam` to `torch.optim.SGD(..., lr=0.01, momentum=0.9)`. Compare epoch-1 loss.
3. **Batch size:** Train with `--batch-size 16` and `--batch-size 256`. What happens to time per epoch and how noisy the loss looks?
4. **Dropout:** Insert `nn.Dropout(0.2)` after each ReLU. Watch what happens if you forget `model.eval()` in `evaluate.py`.
5. **Wrong input shape:** In `predict.py`, skip `.unsqueeze(0)` and read the error. Then restore it and explain why the batch dimension is required.

---

## License / data

MNIST is loaded through torchvision. This project does not use pretrained models or TensorFlow/Keras.
