import numpy as np

from utils import plot_regression_1d


def ridge_regression(y, tx, lambda_):
    tx_t = tx.T
    identity = np.eye(tx.shape[1])

    identity[0, 0] = 0
    w = np.linalg.inv(tx_t @ tx + lambda_ * identity) @ tx_t @ y
    return w


def main():
    n, d = 100, 1
    lambda_ = 1

    original_tx = np.random.randn(n, d)

    bias = np.ones((n, 1))
    tx = np.hstack((bias, original_tx))

    w_true = np.random.randn(d + 1)

    noise = np.random.randn(n)

    y = tx @ w_true + noise

    w_learned = ridge_regression(y, tx, lambda_)

    print("Learned weights (including bias):", w_learned)
    print("True weights (including bias):", w_true)

    plot_regression_1d(tx, y, w_learned)


if __name__ == "__main__":
    main()
