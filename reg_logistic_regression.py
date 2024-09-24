import numpy as np

from logistic_regression import sigmoid, log_reg_grad, log_reg_loss, logistic_regression




""" Tests """
MAX_ITERS = 2
GAMMA = 0.1
RTOL = 1e-4
ATOL = 1e-8


def test_reg_logistic_regression_0_step(reg_log_reg_func, y, tx):
    lambda_ = 1.0
    expected_w = np.array([0.409111, 0.843996])
    y = (y > 0.2) * 1.0
    w, loss = reg_log_reg_func(
        y, tx, lambda_, expected_w, 0, GAMMA
    )

    expected_loss = 1.407327

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_reg_logistic_regression(reg_log_reg_func, y, tx, initial_w):
    lambda_ = 1.0
    y = (y > 0.2) * 1.0
    w, loss = reg_log_reg_func(
        y, tx, lambda_, initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 0.972165
    expected_w = np.array([0.216062, 0.467747])

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
    test_reg_logistic_regression_0_step(reg_logistic_regression, y, tx)
    test_reg_logistic_regression(reg_logistic_regression, y, tx, initial_w)

    print("Tests passed, good job!")


if __name__ == "__main__":
    main()
