import numpy as np

from utils import plot_regression_1d


def least_squares(y, tx):
    """
    Computes the least squares solution to the linear regression problem.

    Parameters:
    y (np.ndarray): Target values, shape (n_samples,)
    tx (np.ndarray): Feature matrix, shape (n_samples, n_features)

    Returns:
    w (np.ndarray): Optimal weights, shape (n_features,)
    loss (float): Mean Squared Error loss
    """
    tx_t = tx.T
    w = np.linalg.solve(tx_t @ tx, tx_t @ y)

    # Compute Mean Squared Error for the Loss
    loss = np.square(y - tx @ w).mean()
    return w, loss


def main():
    n, d = 100, 1

    original_tx = np.random.randn(n, d)

    # Sampling the data from a normal distribution and inserting the bias as the first dimension of each datum
    bias = np.ones((n, 1))
    tx = np.hstack((bias, original_tx))

    # Sampling the true weights from a normal distribution with d + 1 dimensions to account for the bias
    w_true = np.random.randn(d + 1)

    noise = np.random.randn(n)

    y = tx @ w_true + noise

    w_learned, loss = least_squares(y, tx)

    print("Learned weights (including bias):", w_learned)
    print("True weights (including bias):", w_true)
    print("Loss when using the learned weights:", loss)

    if d == 1:
        plot_regression_1d(tx, y, w_learned)


if __name__ == "__main__":
    main()
