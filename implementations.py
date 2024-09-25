import numpy as np
from helpers import batch_iter
from logistic_regression import sigmoid, log_reg_grad, log_reg_loss, logistic_regression


def loss_mse(y, tx, w):
    """Calculate the mean squared error loss.

    Parameters:
    y (np.ndarray): Target values, shape (n_samples,)
    tx (np.ndarray): Feature matrix, shape (n_samples, n_features)
    w (np.ndarray): Weights, shape (n_features,)

    Returns:
    float: Mean Squared Error loss
    """
    return 1/2 * np.square(y - tx @ w).mean()

def compute_gradient_linreg(y, tx, w):
    """
    Calculate the gradient of the square mean error loss function of a linear regression model, with respect to the weights w.

    Parameters:
        y : np.ndarray(N) : labels (0 or 1)
        tx : np.ndarray(N, D) : features, including the constant feature
        w : np.ndarray(D) : weights

    Returns:
        g: np.ndarray(D) : gradient of the loss with respect to w
    """
    return -1/len(y) * np.transpose(tx) @ (y - tx @ w) 


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
    w = initial_w
    batches = batch_iter(y, tx, 1, max_iters)

    for n_iter, batch in enumerate(batches):
        y_batch, xt_batch = batch
        g = compute_gradient_linreg(y_batch, xt_batch, w)
        w = w - gamma * g
    return w, loss_mse(y, tx, w)

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
    w = initial_w

    for n_iter in range(max_iters):
        g = compute_gradient_linreg(y, tx, w)
        w = w - gamma * g
    return w, loss_mse(y, tx, w)

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
    return w, loss_mse(y, tx, w)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent

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
    final_loss = log_reg_loss(y, tx, w) # don't include the regularization term

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
    n, d = tx.shape
    w = np.linalg.solve(tx.T @ tx + lambda_ * 2*n * np.eye(d), tx.T @ y)

    # Compute Mean Squared Error for the Loss
    return w, loss_mse(y, tx, w)
