"""Train the digits MLP and save its learning curve and raw numbers (Part 2).

Run: uv run python -m train.train_mlp
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write the figure to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np

from nn import Adam
from train.data import load_digits_data
from train.loop import train
from train.model import build_mlp
from utils import set_seed

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main(seed=0, epochs=30, lr=0.01):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    X_train, y_train, X_test, y_test = load_digits_data(seed=seed)
    model = build_mlp(rng)
    opt = Adam(model.parameters(), lr=lr)
    history = train(model, opt, X_train, y_train, X_test, y_test, epochs=epochs, seed=seed)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "part2_learning_curve.json", "w") as f:
        json.dump({"seed": seed, "epochs": epochs, "lr": lr, **history}, f, indent=2)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(history["loss"])
    ax_loss.set(title="Training loss", xlabel="epoch", ylabel="cross-entropy")
    ax_acc.plot(history["train_acc"], label="train")
    ax_acc.plot(history["test_acc"], label="test")
    ax_acc.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "part2_learning_curve.png", dpi=120)

    print(f"final train acc {history['train_acc'][-1]:.4f}  "
          f"test acc {history['test_acc'][-1]:.4f}")
    print(f"saved {RESULTS / 'part2_learning_curve.png'}")


if __name__ == "__main__":
    main()
