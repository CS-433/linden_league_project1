import numpy as np
from functools import partial
from helpers import batch_iter
from utils import NegFn
from collections import Counter


class LogisticRegression:
    """
    Logistic regression binary classifier

    Parameters:
        init_w (np.ndarray(D) or None) : initial weights
        predict_thres (float in range[0,1]) : threshold for binary classification
        max_iters (int) : maximum number of iterations
        gamma (float) : step size
        use_line_search (bool) : whether to use line search, gamma is ignored if True
        class_weights (dict) : class weights as {class: weight}
        optim_algo (str) : optimization algorithm to use (gd, sgd, newton, snewton, lbfgs, slbfgs)
        optim_kwargs (dict) : optimization algorithm kwargs
        verbose (bool) : verbosity
    """
    def __init__(
        self,
        init_w=None,
        predict_thres=0.5,
        max_iters=1000,
        gamma=1e-4,
        use_line_search=False,
        reg_mul=0,
        class_weights=None,
        optim_algo="gd",
        optim_kwargs=None,
        verbose=False,
    ):
        self.w = None
        self.predict_thres = predict_thres
        self.max_iters = max_iters
        self.gamma = gamma
        self.use_line_search = use_line_search
        self.init_w = init_w
        self.reg_mul = reg_mul
        self.class_weights = class_weights
        self.optim_algo = optim_algo
        self.optim_kwargs = optim_kwargs if optim_kwargs is not None else dict()
        self.direction_fn = {
            "gd": NegFn(self.log_reg_grad),
            "sgd": NegFn(self.log_reg_grad),
            "newton": NegFn(self._get_newton_direction),
            "snewton":NegFn(self._get_newton_direction),
            "lbfgs": None,
            "slbfgs": None,
        }
        self.verbose = verbose

    @staticmethod
    def sigmoid(x):
        """ Sigmoid function

        Parameters:
            x : {float, np.ndarray, int} : input

        Returns:
            {float, np.ndarray} : sigmoid(x)
        """
        return 1. / (1 + np.exp(-x))

    def log_reg_loss(self, x, y, w, sample_weights=None):
        """ Compute the logistic regression loss

        Parameters:
            x : np.ndarray(N, D) : features
            y : np.ndarray(N) : labels (0 or 1)
            w : np.ndarray(D) : weights
            sample_weights : np.ndarray(N) : sample weights (default to uniform)

        Returns:
            loss : float : loss value (negative log likelihood)
        """
        z = x @ w # (N)

        ### negative log likelihood (derived from -y*log(p) - (1-y)*log(1-p))
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0])
        loss = np.sum(
            (np.log(1 + np.exp(z)) - y * z) * sample_weights
        ) / sample_weights.sum()

        ### add regularization
        if self.reg_mul > 0:
            loss += self.reg_mul * np.linalg.norm(w)**2

        return loss

    def log_reg_grad(self, x, y, w, sample_weights=None):
        """ Compute the gradient of the logistic regression loss

        Parameters:
            x : np.ndarray(N, D) : features
            y : np.ndarray(N) : labels (0 or 1)
            w : np.ndarray(D) : weights
            sample_weights : np.ndarray(N) : sample weights (default to uniform)
            reg_mul : float : regularization multiplier

        Returns:
            grad : np.ndarray(D) : gradient
        """
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0])
        probas = LogisticRegression.sigmoid(x @ w) # (N)
        grad = x.T @ (sample_weights * (probas - y)) / sample_weights.sum()

        ### add regularization
        if self.reg_mul > 0:
            grad += 2 * self.reg_mul * w

        return grad

    def log_reg_hessian(self, x, w, sample_weights=None):
        """ Compute the Hessian of the logistic regression loss

        Parameters:
            x : np.ndarray(N, D) : features
            w : np.ndarray(D) : weights
            sample_weights : np.ndarray(N) : sample weights (default to uniform)

        Returns:
            hess : np.ndarray(D, D) : Hessian
        """
        if sample_weights is None:
            sample_weights = np.ones(x.shape[0])
        probas = LogisticRegression.sigmoid(x @ w)
        diag = probas * (1 - probas) * sample_weights
        hess = x.T @ np.diag(diag) @ x / sample_weights.sum()

        ### add regularization
        if self.reg_mul > 0:
            hess += 2 * self.reg_mul * np.eye(x.shape[1])

        return hess

    def _line_search(self, w, direction, loss_fn, grad_fn, gamma_init=1., c=1e-4):
        """ Perform backtracking line search

        Parameters:
            w : np.ndarray(D) : weights
            direction : np.ndarray(D) : direction
            loss_fn : function : loss function
            grad_fn : function : gradient function
            gamma_init : float : initial step size
            c : float : constant for Armijo condition (between 0 and 1)

        Returns:
            gamma : float : step size
        """
        gamma = gamma_init
        while True:
            ### get current loss and new loss
            loss = loss_fn(w=w)
            w_new = w + gamma * direction
            loss_new = loss_fn(w=w_new)
            ### check Armijo–Goldstein condition
            if loss_new <= loss + c * gamma * grad_fn(w=w).T @ direction:
                break # step size found

            ### update step size
            gamma *= 0.5
            if gamma < 1e-7:
                break # prevent overflow
        return gamma

    def _lbfgs(self, w, loss_fn, grad_fn, tol=5e-4, m=10, eps=1e-8):
        """ L-BFGS optimization

        Parameters:
            w : np.ndarray(D) : initial weights
            loss_fn : function : loss function
            grad_fn : function : gradient function
            tol : float : tolerance
            m : int : number of history updates to keep
            eps : float : small value to prevent division by zero

        Returns:
            w : np.ndarray(D) : final weights
        """
        ### history of updates
        s_list = [] # s :=  w_{k+1} - w_k
        z_list = [] # z := grad(w_{k+1}) - grad(w_k)
        rho_list = [] # rho := 1 / (z.T @ s)
        g = grad_fn(w=w)
        q = g.copy()

        for k in range(self.max_iters):
            alpha = [] # alpha := rho * s.T @ q
            if k == 0:
                ### first iteration
                direction = -g
            else:
                ### 2-loop recursion to compute the direction d_k = -hess_k^-1 * grad_k
                q = g.copy()
                for i in range(len(s_list) - 1, -1, -1): # iterate backwards
                    alpha_i = rho_list[i] * (s_list[i].T @ q)
                    alpha.append(alpha_i)
                    q -= alpha_i * z_list[i]

                ### scale the Hessian approximation
                if len(z_list) > 0:
                    gamma = (s_list[-1].T @ z_list[-1]) / (z_list[-1].T @ z_list[-1] + eps)
                else:
                    gamma = 1.0
                r = gamma * q

                for i in range(len(s_list)): # iterate forwards to apply the corrections
                    beta = rho_list[i] * (z_list[i].T @ r)
                    r += s_list[i] * (alpha[len(s_list) - 1 - i] - beta)

                direction = -r

            ### get new weights
            if self.use_line_search:
                step_size = self._line_search(w=w, direction=direction, loss_fn=loss_fn, grad_fn=grad_fn)
            else:
                step_size = self.gamma
            s = step_size * direction
            w += s

            ### update gradient
            g_next = grad_fn(w=w)
            z = g_next - g
            g = g_next

            ### update history if the condition on curvature is satisfied
            if s.T @ z > 1e-8:
                if len(s_list) == m: # limit the history size
                    s_list.pop(0)
                    z_list.pop(0)
                    rho_list.pop(0)
                s_list.append(s)
                z_list.append(z)
                rho_list.append(1.0 / (z.T @ s + eps))

            ### check convergence
            if np.linalg.norm(g, ord=2) < tol:
                break

        return w

    def _init_w(self, D):
        """ Initialize weights

        Parameters:
            D : int : number of features

        Returns:
            w : np.ndarray(D) : initial weights

        """
        if self.init_w is not None:
            return self.init_w.copy()
        return np.zeros(D)

    def _iter_optimize(self, w, dataloader, direction_fn, direction_fn_kwargs=None, use_lbfgs=False):
        """ Run iterative optimization

        Parameters:
            w : np.ndarray(D) : initial weights
            dataloader : list : list of (x, y) pairs (can be mulltidimensional)
            direction_fn : function : function to compute the direction
            direction_fn_kwargs : dict : kwargs for the direction function
            use_lbfgs : bool : whether to use L-BFGS

        Returns:
            w : np.ndarray(D) : final weights
        """
        for (x, y) in dataloader:
            ### set sample weights from class weights
            self.sample_weights = np.ones(y.shape[0])
            if self.class_weights is not None:
                self.sample_weights = np.array([self.class_weights[yi] for yi in y])

            ### update weights
            if not use_lbfgs:
                direction = direction_fn(x=x, y=y, w=w, sample_weights=self.sample_weights, **direction_fn_kwargs or dict())
                step_size = self.gamma
                if self.use_line_search:
                    step_size = self._line_search(
                        w=w,
                        direction=direction,
                        loss_fn=partial(self.log_reg_loss, x=x, y=y, sample_weights=self.sample_weights),
                        grad_fn=partial(self.log_reg_grad, x=x, y=y, sample_weights=self.sample_weights)
                    )
                w += step_size * direction
            else:
                w = self._lbfgs(
                    w=w,
                    loss_fn=partial(self.log_reg_loss, x=x, y=y, sample_weights=self.sample_weights),
                    grad_fn=partial(self.log_reg_grad, x=x, y=y, sample_weights=self.sample_weights),
                    tol=self.optim_kwargs.get("tol", 5e-4),
                    m=self.optim_kwargs.get("m", 30),
                    eps=self.optim_kwargs.get("eps", 1e-8),
                )
        return w

    def _get_newton_direction(self, x, y, w, sample_weights=None):
        """ Compute the Newton direction

        Parameters:
            x : np.ndarray(N, D) : features
            y : np.ndarray(N) : labels (0 or 1)
            w : np.ndarray(D) : weights
            sample_weights : np.ndarray(N) : sample weights (default to uniform)

        Returns:
            direction : np.ndarray(D) : Newton direction
        """
        g = self.log_reg_grad(x=x, y=y, w=w, sample_weights=sample_weights)
        hess = self.log_reg_hessian(x=x, w=w, sample_weights=sample_weights)
        return np.linalg.solve(hess, g)

    def fit(self, x, y):
        """ Fit the model to the data

        Parameters:
            x : np.ndarray(N, D) : features
            y : np.ndarray(N) : labels (0 or 1)

        Returns:
            w : np.ndarray(D) : final weights
            loss : float : final loss
        """
        assert np.isnan(x).sum() == 0, "NaN values in features are not supported"
        assert np.isnan(y).sum() == 0, "NaN values in labels are not supported"

        ### initialize weights
        self.w = self._init_w(x.shape[1])

        ### validate class weights
        if self.class_weights is not None:
            assert set(y).issubset(set(self.class_weights.keys())), "Class weights must be provided for all classes"

        ### prepare data
        if self.optim_algo in ("sgd", "snewton", "slbfgs"):
            make_dataloader = lambda: batch_iter(
                y=y,
                tx=x,
                batch_size=self.optim_kwargs.get("batch_size", 1),
                num_batches=self.optim_kwargs.get("num_batches", x.shape[0] // self.optim_kwargs.get("batch_size", 1)),
                x_first=True,
                shuffle=True,
            )
        else:
            make_dataloader = lambda: [(x, y)]

        ### run optimization
        epochs = self.optim_kwargs.get("epochs", 1)
        for ep in range(epochs):
            dataloader = make_dataloader()
            self.w = self._iter_optimize(
                w=self.w,
                dataloader=dataloader,
                direction_fn=self.direction_fn[self.optim_algo],
                use_lbfgs="lbfgs" in self.optim_algo,
            )
            if self.verbose:
                print(f"[Epoch {ep}/{epochs}] Loss: {self.log_reg_loss(x=x, y=y, w=self.w)}")

        return self

    def predict(self, x):
        """ Predict the labels of the data

        Parameters:
            x : np.ndarray(N, D) : features

        Returns:
            y_pred : np.ndarray(N) : predicted labels
        """
        if self.w is None:
            raise ValueError("Model has not been trained yet")

        probas = LogisticRegression.sigmoid(x @ self.w)
        y_pred = (probas > self.predict_thres).astype(int)
        return y_pred

    def predict_proba(self, x):
        """ Predict the probabilities of the data

        Parameters:
            x : np.ndarray(N, D) : features

        Returns:
            probas : np.ndarray(N) : predicted probabilities
        """
        if self.w is None:
            raise ValueError("Model has not been trained yet")

        probas = LogisticRegression.sigmoid(x @ self.w)
        return probas


class DecisionTreeBinaryClassifier:
    """
    Decision tree binary classifier

    Parameters:
        max_depth (int) : maximum depth of the tree
        min_samples_split (int) : minimum number of samples required to split a node
        criterion (str) : impurity criterion (gini or entropy)
        class_weights (dict) : class weights as {class: weight}
        eval_max_n_thresholds_per_split (int) : maximum number of thresholds to evaluate per feature
    """
    def __init__(
        self,
        max_depth=None,
        min_samples_split=5,
        criterion="gini",
        class_weights=None,
        eval_max_n_thresholds_per_split=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.class_weights = class_weights
        self.eval_max_n_thresholds_per_split = eval_max_n_thresholds_per_split # for faster building of the tree
        self.tree = None

    class Node:
        """ Node class for the decision tree """
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature  # index of the feature to split on
            self.threshold = threshold  # threshold value for the split
            self.left = left  # left child node
            self.right = right  # right child node
            self.value = value  # leaf value for prediction

    def _gini(self, y):
        """ Compute Gini impurity for labels y

        Parameters:
            y : np.ndarray(N) : labels

        Returns:
            gini_impurity : float : Gini impurity
            info : dict : additional info
        """
        N = len(y)
        if N == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)

        ### compute (weighted) Gini impurity
        weights = np.ones_like(counts, dtype=float)
        if self.class_weights is not None:
            weights = np.array([self.class_weights[c] for c in classes])
        probas = (counts * weights) / (counts * weights).sum()
        gini_impurity = 1 - np.sum(probas**2)

        return gini_impurity, {"counts": counts, "probas": probas, "weights": weights}

    def _entropy(self, y):
        """ Compute entropy for labels y

        Parameters:
            y : np.ndarray(N) : labels

        Returns:
            entropy : float : entropy
            info : dict : additional info
        """
        N = len(y)
        if N == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)

        weights = np.ones_like(counts, dtype=float)
        if self.class_weights is not None:
            weights = np.array([self.class_weights[c] for c in classes])

        probas = (counts * weights) / (counts * weights).sum()
        entropy = -np.sum([p * np.log2(p) for p in probas if p > 0])
        return entropy, {"counts": counts, "probas": probas, "weights": weights}

    def _calculate_impurity(self, y):
        """ Compute the impurity measure (Gini or Entropy)

        Parameters:
            y : np.ndarray(N) : labels

        Returns:
            impurity : float : impurity
            info : dict : additional info
        """
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        else:
            raise ValueError("Unknown criterion. Use 'gini' or 'entropy'.")

    def _split(self, x, y, feature, threshold):
        """ Split the dataset based on a feature and threshold

        Parameters:
            x : np.ndarray(N, D) : features
            y : np.ndarray(N) : labels
            feature : int : feature index
            threshold : float : threshold value

        Returns:
            x_left : np.ndarray(N, D) : left split features
            x_right : np.ndarray(N, D) : right split features
            y_left : np.ndarray(N) : left split labels
            y_right : np.ndarray(N) : right split labels
        """
        left_mask = x[:, feature] <= threshold
        right_mask = x[:, feature] > threshold
        return x[left_mask], x[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, x, y):
        """ Find the best feature and threshold to split on

        Parameters:
            x : np.ndarray(N, D) : features
            y : np.ndarray(N) : labels

        Returns:
            best_feature : int : best feature index
            best_threshold : float : best threshold value
        """
        N, D = x.shape
        if N <= 1:
            return None, None

        ### find the best split
        best_impurity = self._calculate_impurity(y)[0]
        best_feature, best_threshold = None, None

        for feature in range(D):
            ### find the best threshold (split value) for the current feature
            thresholds = np.unique(x[:, feature])
            if self.eval_max_n_thresholds_per_split is not None:
                thresholds = thresholds[np.linspace(0, len(thresholds) - 1, self.eval_max_n_thresholds_per_split, dtype=int)]
            for threshold in thresholds:
                x_left, x_right, y_left, y_right = self._split(x, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0: # skip invalid splits
                    continue

                ### weighted average of the split impurities (incorporating class weights if provided)
                left_impurity, left_info = self._calculate_impurity(y_left)
                right_impurity, right_info = self._calculate_impurity(y_right)
                if self.class_weights:
                    left_weighted_n_samples = (left_info["counts"] * left_info["weights"]).sum()
                    right_weighted_n_samples = (right_info["counts"] * right_info["weights"]).sum()
                else:
                    left_weighted_n_samples = len(y_left)
                    right_weighted_n_samples = len(y_right)
                impurity = (
                    left_weighted_n_samples * left_impurity
                    + right_weighted_n_samples * right_impurity
                ) / (left_weighted_n_samples + right_weighted_n_samples)

                ### update best split
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, x, y, depth=0):
        """ Recursively build the decision tree

        Parameters:
            x : np.ndarray(N, D) : features
            y : np.ndarray(N) : labels
            depth : int : current depth

        Returns:
            node : Node : root node of the tree
        """
        ### check stopping criteria (leaf node)
        if (x.shape[0] < self.min_samples_split
            or len(np.unique(y)) == 1
            or depth == self.max_depth):
            leaf_value = self._most_common_label(y) # predict the most common label
            return self.Node(value=leaf_value)

        ### find the best split
        feature, threshold = self._best_split(x, y)
        if feature is None: # no improvement from splitting, stop
            return self.Node(value=self._most_common_label(y))
        x_left, x_right, y_left, y_right = self._split(x, y, feature, threshold)

        ### recursively build the tree from the left and right splits
        left_child = self._build_tree(x_left, y_left, depth + 1)
        right_child = self._build_tree(x_right, y_right, depth + 1)

        return self.Node(feature=feature, threshold=threshold, left=left_child, right=right_child)

    def _most_common_label(self, y):
        """ Return the most common label in y

        Parameters:
            y : np.ndarray(N) : labels

        Returns:
            most_common : int : most common label
        """
        classes, counts = np.unique(y, return_counts=True)

        ### apply class weights
        if self.class_weights is not None:
            counts = np.array([counts[i] * self.class_weights[c] for i, c in enumerate(classes)])

        most_common = classes[np.argmax(counts)]
        return most_common

    def fit(self, x, y):
        """ Fit the decision tree to the dataset

        Parameters:
            x : np.ndarray(N, D) : features
            y : np.ndarray(N) : labels

        Returns:
            tree : Node : root node of the tree
        """
        assert np.isnan(x).sum() == 0, "NaN values in features are not supported"
        assert np.isnan(y).sum() == 0, "NaN values in labels are not supported"

        self.tree = self._build_tree(np.array(x), np.array(y))

    def _predict(self, x, node):
        """ Recursively predict the label for a single sample

        Parameters:
            x : np.ndarray(D) : features
            node : Node : current node

        Returns:
            value : int : predicted label
        """
        ### check if we reached a leaf node
        if node.value is not None:
            return node.value

        ### recursively traverse the tree (left/right)
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    def predict(self, x):
        """ Predict the labels for all samples in x

        Parameters:
            x : np.ndarray(N, D) : features

        Returns:
            y_pred : np.ndarray(N) : predicted labels
        """
        x = np.array(x)
        return np.array([self._predict(_x, self.tree) for _x in x])


class PCA:
    """
    Principal Component Analysis (PCA)

    Parameters:
        n_components (int) : number of principal components to keep
        min_explained_variance (float) : minimum fraction of explained variance
        standardize (bool) : whether to standardize features before applying PCA
    """
    def __init__(self, n_components=None, min_explained_variance=None, standardize=True):
        assert not (n_components and min_explained_variance), \
            "Provide either n_components or min_explained_variance, but not both."
        self.n_components = n_components
        self.min_explained_variance = min_explained_variance
        self.standardize = standardize
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def _standardize(self, x):
        """ Standardize the dataset by removing the mean and scaling to unit variance

        Parameters:
            x : np.ndarray(N, D) : data matrix

        Returns:
            x_standardized : np.ndarray(N, D) : standardized data
        """
        self.mean_ = np.mean(x, axis=0)
        x_centered = x - self.mean_
        x_std = np.std(x_centered, axis=0, ddof=1)
        x_standardized = x_centered / (x_std + 1e-7)
        return x_standardized

    def _covariance_matrix(self, x):
        """ Compute the covariance matrix of the dataset

        Parameters:
            x : np.ndarray(N, D) : data matrix

        Returns:
            cov_matrix : np.ndarray(D, D) : covariance matrix
        """
        cov_matrix = (x.T @ x) / (x.shape[0] - 1)
        return cov_matrix

    def fit(self, x):
        """ Fit the PCA model to the dataset

        Parameters:
            x : np.ndarray(N, D) : data matrix

        Returns:
            self : PCA : fitted PCA instance
        """
        ### standardize the data
        if self.standardize:
            x = self._standardize(x)

        ### compute the covariance matrix
        cov_matrix = self._covariance_matrix(x)

        ### compute the eigendecomposition of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov_matrix) # eigenvectors in columns

        ### sort the eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        ### keep only selected number of components
        if self.n_components is not None: # keep only the top n components
            ### keep only the top n components
            eigvals = eigvals[:self.n_components]
            self.components_ = eigvecs[:, :self.n_components]
        elif self.min_explained_variance is not None:
            ### keep components that explain at least min_explained_variance of the variance
            eigvals = eigvals[~(np.cumsum(eigvals) / (np.sum(eigvals) + 1e-7) > self.min_explained_variance)]
            self.components_ = eigvecs[:, :len(eigvals)]

        ### store explained variance
        self.explained_variance_ = eigvals
        self.explained_variance_ratio_ = eigvals / (np.sum(eigvals) + 1e-7)

        return self

    def transform(self, x):
        """ Project the data onto the principal components

        Parameters:
            x : np.ndarray(N, D) : data matrix

        Returns:
            x_pca : np.ndarray(N, n_components) : transformed data
        """
        ### standardize the data
        if self.standardize:
            x = self._standardize(x)

        ### project the data onto the principal components
        return x @ self.components_

class SVM:
    """
    Support Vector Machine (SVM) binary classifier

    Parameters:
        _lambda (float) : regularization parameter
        max_iters (int) : maximum number of iterations
        gamma (float) : step size
        class_weights (dict) : class weights as {class: weight}
    """
    def __init__(self, _lambda=0.1, max_iters=10_000, gamma=.01, class_weights = {0: 1, 1: 4}):
        self._lambda = _lambda
        self.max_iters = max_iters
        self.gamma = gamma
        self.class_weights = class_weights
    def calculate_coordinate_update(self, x, y, alpha, w, n):
        """compute a coordinate update (closed form) for coordinate n.

        Args:
            y: the corresponding +1 or -1 labels, shape = (num_examples)
            X: the dataset matrix, shape = (num_examples, num_features)
            lambda_: positive scalar number
            alpha: vector of dual coordinates, shape = (num_examples)
            w: vector of primal parameters, shape = (num_features)
            n: the coordinate to be updated

        Returns:
            w: updated vector of primal parameters, shape = (num_features)
            alpha: updated vector of dual parameters, shape = (num_examples)

        >>> y_test = np.array([1, -1])
        >>> x_test = np.array([[1., 2., 3.], [4., 5., 6.]])
        >>> w_test = np.array([-0.3, -0.3, -0.3])
        >>> alpha_test = np.array([0.1, 0.1])
        >>> calculate_coordinate_update(y_test, x_test, 1, alpha_test, w_test, 0)
        (array([-0.1,  0.1,  0.3]), array([0.5, 0.1]))
        """
        # calculate the update of coordinate at index=n.
        N = y.size
        x_n, y_n = x[n], y[n]
        # Convert the 0 or 1 label y_n to -1 or 1
        y_n_prime = 1 if y_n == 1 else -1

        old_alpha_n = np.copy(alpha[n]).item()

        gamma = self._lambda * N * (1- w.dot(x_n) * y_n_prime)/ (np.linalg.norm(x_n)**2 + .00001) # avoid division by zero

        gamma = min(1-old_alpha_n, gamma)
        gamma = max(-old_alpha_n, gamma)
        assert y_n in self.class_weights, f"y_n={y_n} not in class_weights={self.class_weights}"

        alpha[n] += gamma
        w += 1/(self._lambda* N) * gamma * self.class_weights[y_n] * y_n_prime * x_n
        return w, alpha


    def fit(self, x, y):
        """ Fit the SVM to the dataset

        Parameters:
            x : np.ndarray(N, D) : features
            y : np.ndarray(N) : labels

        Returns:
            tree : Node : root node of the tree
        """
        num_examples, num_features = x.shape
        w = np.zeros(num_features)
        alpha = np.zeros(num_examples)

        for it in range(self.max_iters):
            n = np.random.randint(0, num_examples - 1)
            w, alpha = self.calculate_coordinate_update(x, y, alpha, w, n)
        self.w = w

    def predict(self, x):
        """ Predict the labels for all samples in x

        Parameters:
            x : np.ndarray(N, D) : features

        Returns:
            y_pred : np.ndarray(N) : predicted labels
        """
        return (x @ self.w > 0).astype(int)

class KNN:
    def __init__(self, k = 5):
        self.k = k
    
    def fit(self, x, y):
        self.train_x = x
        self.train_y = y

    def knn(self, x):
        square_distances = np.sum((self.train_x - x[np.newaxis, :])**2, axis=1)
        return np.argpartition(square_distances, self.k)[:self.k]


    def predict(self, x):
        closest = self.knn(x)
        counter = Counter(self.train_y[closest])
        mode = max(counter, key=counter.get)
        return mode


    def predict_all(self, x):
        return np.array([self.predict(xi) for xi in x])



