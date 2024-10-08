import os
import random
import itertools
import datetime
import dateutil.tz
import matplotlib.pyplot as plt
import numpy as np


def plot_regression_1d(tx, y, w):
    os.makedirs("results", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    x_feature = tx[:, 1]

    ax.scatter(x_feature, y, color="blue", label="Data Points", alpha=0.6)

    x_min, x_max = x_feature.min() - 1, x_feature.max() + 1
    x_plot = np.linspace(x_min, x_max, 100)

    y_plot = w[0] + w[1] * x_plot

    ax.plot(x_plot, y_plot, color="red", label="Regression Line", linewidth=2)

    ax.set_xlabel("Feature x", fontsize=12)
    ax.set_ylabel("Target y", fontsize=12)
    ax.set_title("1D Least Squares Regression", fontsize=14)

    ax.legend()

    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    plot_path = os.path.join("results", "regression.png")
    fig.savefig(plot_path)

    print(f"Regression plot saved to {plot_path}")


def split_data(x, y, split_frac=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    idxs = np.random.permutation(x.shape[0])
    n_test = int(x.shape[0] * split_frac)
    test_idxs = idxs[:n_test]
    train_idxs = idxs[n_test:]
    return x[train_idxs], x[test_idxs], y[train_idxs], y[test_idxs]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def f1(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def get_cross_val_scores(model, X, y, k_folds=5, scoring_fn=accuracy, shuffle=True):
    scores = []
    for fold_idx in range(k_folds):
        ### create current fold mask
        if shuffle:
            mask = np.random.permutation(X.shape[0]) % k_folds == fold_idx
        else:
            mask = np.arange(X.shape[0]) % k_folds == fold_idx
        X_train, y_train = X[~mask], y[~mask]
        X_test, y_test = X[mask], y[mask]

        ### fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = scoring_fn(y_test, y_pred)
        scores.append(score)
    return np.array(scores)


def prep_hyperparam_search(hyperparam_search):
    hp_names, hp_vals = zip(*hyperparam_search.items())
    return [dict(zip(hp_names, v)) for v in itertools.product(*hp_vals)]


def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
