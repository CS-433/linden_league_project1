import random
import itertools
import datetime
import dateutil.tz
import numpy as np


def split_data(x, y, split_frac=0.2, seed=None):
    """ Split data into train and test sets

    Parameters:
        x : np.ndarray(N, D) : data matrix
        y : np.ndarray(N) : labels
        split_frac : float : fraction of data to use for testing
        seed : int : random seed

    Returns:
        x_train : np.ndarray : training data
        x_test : np.ndarray : testing data
        y_train : np.ndarray : training labels
        y_test : np.ndarray : testing labels
    """
    ### set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    ### shuffle data and split
    idxs = np.random.permutation(x.shape[0])
    n_test = int(x.shape[0] * split_frac)
    test_idxs = idxs[:n_test]
    train_idxs = idxs[n_test:]

    return x[train_idxs], x[test_idxs], y[train_idxs], y[test_idxs]


def accuracy(y_true, y_pred):
    """ Calculate the accuracy of the predictions

    Parameters:
        y_true : np.ndarray(N) : true labels
        y_pred : np.ndarray(N) : predicted labels

    Returns:
        float : accuracy
    """
    return np.mean(y_true == y_pred)


def f1(y_true, y_pred):
    """ Calculate the F1 score of the predictions

    Parameters:
        y_true : np.ndarray(N) : true labels
        y_pred : np.ndarray(N) : predicted labels

    Returns:
        float : F1 score
    """
    uniq_vals = np.unique(y_true) # sorted
    assert len(uniq_vals) == 2, "F1 score is only implemented for binary classification"
    assert set(y_pred).issubset(uniq_vals), "Predictions contain values not present in true labels"

    ### calculate true positives, false positives, and false negatives
    tp = ((y_true == uniq_vals[1]) & (y_pred == uniq_vals[1])).sum()
    fp = ((y_true == uniq_vals[0]) & (y_pred == uniq_vals[1])).sum()
    fn = ((y_true == uniq_vals[1]) & (y_pred == uniq_vals[0])).sum()

    ### calculate precision, recall, and F1
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1


def get_cross_val_scores(model, X, y, k_folds=5, scoring_fn=accuracy, shuffle=True):
    """ Perform k-fold cross-validation and return the scores

    Parameters:
        model : object : model to evaluate
        X : np.ndarray(N, D) : features
        y : np.ndarray(N) : labels
        k_folds : int : number of folds
        scoring_fn : function : scoring function
        shuffle : bool : shuffle data before splitting

    Returns:
        scores : np.ndarray(k_folds) : scores for each fold
    """
    scores = []
    for fold_idx in range(k_folds):
        ### create current fold mask and split data
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
    """ Prepare hyperparameter search space

    Parameters:
        hyperparam_search : dict : hyperparameter search space (each key is a hyperparameter name and value is a list of values)

    Returns:
        list : list of hyperparameter combinations
    """
    hp_names, hp_vals = zip(*hyperparam_search.items())
    return [dict(zip(hp_names, v)) for v in itertools.product(*hp_vals)]


def now_str():
    """ Get the current date and time as a string

    Returns:
        str : current date and time as a string
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def seed_all(seed):
    """ Seed all random number generators

    Parameters:
        seed : int : seed
    """
    random.seed(seed)
    np.random.seed(seed)


class NegFn:
    """
    Wrapper to negate the output of a function

    Parameters:
        fn : function : function to negate
    """
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return -self.fn(*args, **kwargs)
