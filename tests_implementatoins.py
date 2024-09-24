import numpy as np

from utils import plot_regression_1d


def test_least_squares():
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


def logistic_regression_test():
    ### data
    y = np.array([0.1, 0.3, 0.5])
    tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
    initial_w = np.array([0.5, 1.0])

    ### tests
    test_logistic_regression_0_step(logistic_regression, y, tx)
    test_logistic_regression(logistic_regression, y, tx, initial_w)

    print("Tests passed, good job!")


def ridge_regression_test():
    n, d = 100, 1
    lambda_ = 1

    original_tx = np.random.randn(n, d)

    # Sampling the data from a normal distribution and inserting the bias as the first dimension of each datum
    bias = np.ones((n, 1))
    tx = np.hstack((bias, original_tx))

    # Sampling the true weights from a normal distribution with d + 1 dimensions to account for the bias
    w_true = np.random.randn(d + 1)

    noise = np.random.randn(n)

    y = tx @ w_true + noise

    w_learned, loss = ridge_regression(y, tx, lambda_)

    print("Learned weights (including bias):", w_learned)
    print("True weights (including bias):", w_true)
    print("Loss when using the learned weights:", loss)

    if d == 1:
        plot_regression_1d(tx, y, w_learned)


if __name__ == "__main__":
    pass
