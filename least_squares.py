import numpy as np

from utils import plot_regression_1d


def least_squares(y, tx):
    tx_t = np.transpose(tx)
    return np.linalg.solve(tx_t @ tx, tx_t @ y)


def main():
    n, d = 100, 1

    original_tx = np.random.randn(n, d)

    bias = np.ones((n, 1))
    tx = np.hstack((bias, original_tx))

    w_true = np.random.randn(d + 1)

    noise = np.random.randn(n)

    y = tx @ w_true + noise

    w_learned = least_squares(y, tx)

    print("Learned weights:", w_learned)
    print("True weights:", w_true)

    plot_regression_1d(tx, y, w_learned)


if __name__ == "__main__":
    main()
