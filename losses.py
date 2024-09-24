import numpy as np


def loss_mse(y_true, y_pred):
    return 1 / 2 * np.mean((y_true - y_pred) ** 2)
