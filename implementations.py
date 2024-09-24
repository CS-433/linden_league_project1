import numpy as np
from losses import loss_mse

def compute_gradient(y, tx, w):
    """Compute a gradient at w from a data sample (full sample, or a stochastic batch).
    """
    return -1/len(y) * np.transpose(tx) @ (y - tx @ w) 


def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Perform the given number of iterations of stochastic gradient descent for linear regression using square mean error as loss.
    """
    ws = [initial_w]
    losses = []
    w = initial_w

    batches = batch_iter(y, tx, batch_size, max_iters)

    for n_iter, batch in enumerate(batches):
        y_batch, xt_batch = batch
        g = compute_gradient(y_batch, xt_batch, w)
        loss = loss_mse(y, tx @ w)
        
        w = w - gamma * g
        ws.append(w)
        losses.append(loss)



        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Perform the given number of iterations of gradient descent for linear regression using square mean error as loss.
    """
    ws = [initial_w]
    losses = []
    w = initial_w

    loss = loss_mse(y, tx @ w)
    losses.append(loss)
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

    return losses, ws