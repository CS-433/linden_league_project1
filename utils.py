import os

import matplotlib.pyplot as plt
import numpy as np


def plot_regression_1d(tx, y, w):
    os.makedirs("results", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    x_feature = tx[:, 1]

    ax.scatter(x_feature, y, color="blue", label="Data Points", alpha=0.6)

    x_min, x_max = x_feature.min() - 1, x_feature.max() + 1
    x_plot = np.linspace(x_min, x_max, 100)

    y_plot = w[0] + w[1] * x_plot

    ax.plot(x_plot, y_plot, color="red", label="Regression Line", linewidth=2)

    ax.set_xlabel("Feature x", fontsize=12)
    ax.set_ylabel("Target y", fontsize=12)
    ax.set_title("1D Least Squares Regression", fontsize=14)

    ax.legend()

    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    plot_path = os.path.join("results", "regression.png")
    fig.savefig(plot_path)

    print(f"Regression plot saved to {plot_path}")
