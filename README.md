# MNIST Number Classifier (Keras, minimal)

A tiny Keras/TensorFlow project: trains a model on **MNIST** and prints **only the recognized digit** for a chosen sample or a custom image.

## Features
- Trains a simple MLP model on MNIST (3 epochs by default).
- Predicts on a single MNIST test sample or on your own image.
- Prints only the digit (0–9) with no extra logs.
- (Optional) Quick preview of 25 training images.

## Requirements
- Python 3.9–3.12
- Packages: `tensorflow` (or `tensorflow-cpu`), `numpy`, `Pillow` (for custom images), `matplotlib` (optional for visualization)

### Setup
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS / Linux
    source .venv/bin/activate

    pip install --upgrade pip
    pip install -r requirements.txt

## Project Structure
    .
    ├─ train_and_save.py        # one-time training and saving the model
    ├─ predict_number.py        # loads the model and prints ONLY the digit
    ├─ main_minimal.py          # all-in-one: trains and immediately prints a digit
    ├─ README.md
    ├─ requirements.txt
    └─ .gitignore

## Quick Start (all-in-one)
    python main_minimal.py
Expected output — a single digit (e.g., 2).

## Save-and-Predict Workflow
1) Train and save:
    python train_and_save.py

2) Predict and print only the digit:
    # by MNIST test index
    python predict_number.py --idx 2

    # by your own image (any size — will be resized to 28×28)
    python predict_number.py --image path/to/digit.png

    # if digit is black on white background and results look odd — try inverting
    python predict_number.py --image path/to/digit.png --invert

## (Optional) Preview 25 training images
Add this to any script if you want quick visualization:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([]); plt.yticks([])
        plt.imshow(x_train[i], cmap='gray')
    plt.tight_layout()
    plt.show()

## Notes
- Reduce TensorFlow logs: `import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"`
- Works on CPU; GPU is not required.
- For custom PNG/JPG ensure: normalization (÷255), size (28×28), correct polarity (`--invert` if needed).

## License
MIT — do what you want, no warranty.

===== FILE: requirements.txt =====
tensorflow>=2.14
numpy>=1.24
pillow>=10.0
matplotlib>=3.8
