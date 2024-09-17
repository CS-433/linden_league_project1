import numpy as np


def sigmoid(x):
    """ Sigmoid function

    Parameters:
        x : {float, np.ndarray, int} : input

    Returns:
        {float, np.ndarray} : sigmoid(x)
    """
    return 1. / (1 + np.exp(-x))


def log_reg_grad(y, tx, w):
    """ Compute the gradient of the logistic regression loss

    Parameters:
        y : np.ndarray(N) : labels (0 or 1)
        tx : np.ndarray(N, D) : features
        w : np.ndarray(D) : weights

    Returns:
        grad : np.ndarray(D) : gradient
    """
    N = tx.shape[0]
    probas = sigmoid(tx @ w) # (N)
    grad = tx.T @ (probas - y) / N # (D)
    return grad


def log_reg_loss(y, tx, w):
    """ Compute the logistic regression loss

    Parameters:
        y : np.ndarray(N) : labels (0 or 1)
        tx : np.ndarray(N, D) : features
        w : np.ndarray(D) : weights

    Returns:
        loss : float : loss value (negative log likelihood)
    """
    N = tx.shape[0]
    z = tx @ w # (N)

    ### negative log likelihood (derived from -y*log(p) - (1-y)*log(1-p))
    loss = np.sum(
        np.log(1 + np.exp(z)) - y * z
    ) / tx.shape[0]

    return loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent

    Parameters:
        y : np.ndarray(N) : labels (0 or 1)
        tx : np.ndarray(N, D) : features
        initial_w : np.ndarray(D) : initial weights
        max_iters : int : maximum number of iterations
        gamma : float : step size

    Returns:
        w : np.ndarray(D) : final weights
        loss : float : final loss
    """
    w = initial_w
    for _ in range(max_iters):
        ### compute grad and update w
        grad = log_reg_grad(y, tx, w)
        w = w - gamma * grad
    final_loss = log_reg_loss(y, tx, w)

    return w, final_loss


""" Tests """
MAX_ITERS = 2
GAMMA = 0.1
RTOL = 1e-4
ATOL = 1e-8


def test_logistic_regression_0_step(log_reg_func, y, tx):
    expected_w = np.array([0.463156, 0.939874])
    y = (y > 0.2) * 1.0
    w, loss = log_reg_func(y, tx, expected_w, 0, GAMMA)

    expected_loss = 1.533694

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_logistic_regression(log_reg_func, y, tx, initial_w):
    y = (y > 0.2) * 1.0
    w, loss = log_reg_func(
        y, tx, initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 1.348358
    expected_w = np.array([0.378561, 0.801131])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def main():
    ### data
    y = np.array([0.1, 0.3, 0.5])
    tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
    initial_w = np.array([0.5, 1.0])

    ### tests
    test_logistic_regression_0_step(logistic_regression, y, tx)
    test_logistic_regression(logistic_regression, y, tx, initial_w)

    print("Tests passed, good job!")


if __name__ == "__main__":
    main()
