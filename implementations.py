import numpy as np
from losses import loss_mse
from helpers import batch_iter
from logistic_regression import sigmoid, log_reg_grad, log_reg_loss, logistic_regression


def compute_gradient(y, tx, w):
    """
    Calculate the gradient of the square mean error loss function of a linear regression model, with respect to the weights w.

    Parameters:
        y : np.ndarray(N) : labels (0 or 1)
        tx : np.ndarray(N, D) : features, including the constant feature
        w : np.ndarray(D) : weights

    Returns:
        g: np.ndarray(D) : gradient of the loss with respect to w
    """
    return -1 / len(y) * np.transpose(tx) @ (y - tx @ w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Perform the given number of iterations of stochastic gradient descent for linear regression using square mean error as loss.

    Parameters:
        y : np.ndarray(N) : labels (0 or 1)
        tx : np.ndarray(N, D) : features, including the constant feature
        initial_w : np.ndarray(D) : initial weights
        max_iters : int : maximum number of iterations
        gamma : float : step size

    Returns:
        w : np.ndarray(D) : final parameter vector
        loss : float : final loss
    """

    ws = [initial_w]
    w = initial_w

    losses = [loss_mse(y, tx @ w)]

    batches = batch_iter(y, tx, 1, max_iters)

    for n_iter, batch in enumerate(batches):
        y_batch, xt_batch = batch
        g = compute_gradient(y_batch, xt_batch, w)

        w = w - gamma * g
        ws.append(w)

        loss = loss_mse(y, tx @ w)
        losses.append(loss)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return ws[-1], losses[-1]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Perform the given number of iterations of gradient descent for linear regression using square mean error as loss.

    Parameters:
        y : np.ndarray(N) : labels (0 or 1)
        tx : np.ndarray(N, D) : features, including the constant feature
        initial_w : np.ndarray(D) : initial weights
        max_iters : int : maximum number of iterations
        gamma : float : step size

    Returns:
        w : np.ndarray(D) : final parameter vector
        loss : float : final loss
    """
    ws = [initial_w]
    w = initial_w
    losses = [loss_mse(y, tx @ w)]

    for n_iter in range(max_iters):
        g = compute_gradient(y, tx, w)
        w = w - gamma * g
        loss = loss_mse(y, tx @ w)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return ws[-1], losses[-1]


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
    loss = 1 / 2 * np.square(y - tx @ w).mean()
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent

    Parameters:
        y : np.ndarray(N) : labels (0 or 1)
        tx : np.ndarray(N, D) : features
        lambda_ : float : regularization strength (L_r = \lambda * ||w||^2)
        initial_w : np.ndarray(D) : initial weights
        max_iters : int : maximum number of iterations
        gamma : float : step size

    Returns:
        w : np.ndarray(D) : final parameter vector
        loss : float : final loss
    """
    w = initial_w
    for _ in range(max_iters):
        ### compute grad and update w
        grad = log_reg_grad(y, tx, w) + 2 * lambda_ * w
        w -= gamma * grad
    final_loss = log_reg_loss(y, tx, w)  # don't include the regularization term

    return w, final_loss


def ridge_regression(y, tx, lambda_):
    """
    Computes the least squares solution to the linear regression problem with Ridge regularization.

    Parameters:
    y (np.ndarray): Target values, shape (n_samples,)
    tx (np.ndarray): Feature matrix, shape (n_samples, n_features)
    lambda_ (float): Regularization parameter. Set to 0 for no regularization.

    Returns:
    w (np.ndarray): Optimal weights, shape (n_features,)
    loss (float): Mean Squared Error loss
    """
    tx_t = tx.T
    identity = np.eye(tx.shape[1])

    # The bias should not be affected by regularization term
    identity[0, 0] = 0
    w = np.linalg.solve(tx_t @ tx + lambda_ * identity, tx_t @ y)

    # Compute Mean Squared Error for the Loss
    loss = np.square(y - tx @ w).mean()
    return w, loss
