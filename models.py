import numpy as np
from functools import partial
from helpers import batch_iter
from utils import NegFn


class RidgeRegression:
    """
    Ridge Regression model using the closed-form least squares solution with L2 regularization.

    Parameters:
        reg_mul (float) : regularization multiplier (lambda parameter)
        class_weights (dict) : class weights as {class: weight}
    """

    def __init__(self, reg_mul=0.0, class_weights=None):
        self.reg_mul = reg_mul
        self.class_weights = class_weights
        self.w = None

    def fit(self, x, y):
        """
        Fit the Ridge regression model to the data using the closed-form solution.

        Parameters:
            x : np.ndarray of shape (n_samples, n_features) : Feature matrix
            y : np.ndarray of shape (n_samples,) : Target values (0 or 1)
        """
        ### set sample weights from class weights
        sample_weights = np.ones(y.shape[0])
        if self.class_weights is not None:
            sample_weights = np.array([self.class_weights[yi] for yi in y])

        ### normal equation with sample weights omega: w = (X^T @ (omega * X) + lambda * I)^-1 X^T @ (omega * y)
        try:
            self.w = np.linalg.solve(
                x.T @ (sample_weights[...,None] * x) + self.reg_mul * np.eye(x.shape[1]), 
                x.T @ (sample_weights * y)
            )
        except np.linalg.LinAlgError:
            print("[RidgeRegression] Singular matrix: could not compute the closed-form solution. Setting weights to zeros.")
            self.w = np.zeros(x.shape[1])
        return self

    def predict(self, x, binarize=True):
        """
        Predict target values using the fitted model.

        Parameters:
            x : np.ndarray of shape (n_samples, n_features) : Feature matrix
            binarize : bool : whether to binarize the predictions to 0 or 1

        Returns:
            np.ndarray : Predicted values
        """
        if self.w is None:
            raise ValueError("Model has not been trained yet")

        if binarize:
            return (x @ self.w > 0.5).astype(int)
        return x @ self.w


class LogisticRegression:
    """
    Logistic regression binary classifier

    Parameters:
        init_w (np.ndarray(D), str or None) : initial weights (random, ones, zeros, or custom)
        predict_thres (float in range[0,1]) : threshold for binary classification
        max_iters (int) : maximum number of iterations
        gamma (float) : step size
        use_line_search (bool) : whether to use line search, gamma is ignored if True
        reg_mul (float) : regularization multiplier
        class_weights (dict) : class weights as {class: weight}
        optim_algo (str) : optimization algorithm to use (gd, sgd, lbfgs, slbfgs)
        optim_kwargs (dict) : optimization algorithm kwargs
        update_callback (function) : callback function to call after each parameter update
        verbose (bool) : verbosity
    """
    def __init__(
        self,
        init_w=None,
        predict_thres=0.5,
        max_iters=300,
        gamma=1e-4,
        use_line_search=False,
        reg_mul=0,
        class_weights=None,
        optim_algo="gd",
        optim_kwargs=None,
        update_callback=None,
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
            "lbfgs": None,
            "slbfgs": None,
        }
        self.update_callback = update_callback if update_callback else self.empty_callback
        self.verbose = verbose

    def empty_callback(self, *args, **kwargs):
        """ Empty callback function """
        pass

    @staticmethod
    def sigmoid(x):
        """ Sigmoid function

        Parameters:
            x : {float, np.ndarray, int} : input

        Returns:
            {float, np.ndarray} : sigmoid(x)
        """
        return 1. / (1 + np.exp(np.clip(-x, -50, 50))) # clipping needed for raw-data experiments

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
            (np.log(1 + np.exp(np.clip(z, -50, 50))) - y * z) * sample_weights
        ) / sample_weights.sum() # clipping needed for raw-data experiments

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

            ### check stopping condition
            if loss_new <= loss + c * gamma * grad_fn(w=w).T @ direction:
                break # step size found

            ### update step size
            gamma *= 0.5
            if gamma < 1e-7:
                break # prevent overflow
        return gamma

    def _lbfgs(self, w, loss_fn, grad_fn, tol=5e-4, m=10, eps=1e-8, update_callback_kwargs=None):
        """ L-BFGS optimization

        Parameters:
            w : np.ndarray(D) : initial weights
            loss_fn : function : loss function
            grad_fn : function : gradient function
            tol : float : tolerance
            m : int : number of history updates to keep
            eps : float : small value to prevent division by zero
            update_callback_kwargs : dict : kwargs for the update callback

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
            self.update_callback(w=w, **(update_callback_kwargs or dict()))

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

        if k == self.max_iters - 1 and self.verbose:
            print("Warning: L-BFGS did not converge")

        return w

    def _init_w(self, D):
        """ Initialize weights

        Parameters:
            D : int : number of features

        Returns:
            w : np.ndarray(D) : initial weights

        """
        if self.init_w == "normal":
            return np.random.normal(size=D) * 5e-2
        elif self.init_w == "uniform":
            return np.random.uniform(size=D, low=-np.sqrt(1/D), high=np.sqrt(1/D))
        elif self.init_w == "ones":
            return np.ones(D)
        elif self.init_w == "zeros":
            return np.zeros(D)
        elif type(self.init_w) == np.ndarray:
            return self.init_w.copy()
        return np.zeros(D)

    def _iter_optimize(self, w, dataloader, direction_fn, direction_fn_kwargs=None, use_lbfgs=False, update_callback_kwargs=None):
        """ Run iterative optimization

        Parameters:
            w : np.ndarray(D) : initial weights
            dataloader : list : list of (x, y) pairs (can be mulltidimensional)
            direction_fn : function : function to compute the direction
            direction_fn_kwargs : dict : kwargs for the direction function
            use_lbfgs : bool : whether to use L-BFGS
            update_callback_kwargs : dict : kwargs for the update callback

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
                self.update_callback(w=w, x=x, y=y, direction=direction, step_size=step_size, **(update_callback_kwargs or dict()))
            else:
                w = self._lbfgs(
                    w=w,
                    loss_fn=partial(self.log_reg_loss, x=x, y=y, sample_weights=self.sample_weights),
                    grad_fn=partial(self.log_reg_grad, x=x, y=y, sample_weights=self.sample_weights),
                    tol=self.optim_kwargs.get("tol", 5e-4),
                    m=self.optim_kwargs.get("m", 30),
                    eps=self.optim_kwargs.get("eps", 1e-8),
                    update_callback_kwargs={"x": x, "y": y, **(update_callback_kwargs or dict())}
                )
        return w

    def fit(self, x, y):
        """ Fit the model to the data

        Parameters:
            x : np.ndarray(N, D) : features
            y : np.ndarray(N) : labels (0 or 1)

        Returns:
            self : LogisticRegression : fitted model instance
        """
        assert np.isnan(x).sum() == 0, "NaN values in features are not supported"
        assert np.isnan(y).sum() == 0, "NaN values in labels are not supported"

        ### initialize weights
        self.w = self._init_w(x.shape[1])

        ### validate class weights
        if self.class_weights is not None:
            assert set(y).issubset(set(self.class_weights.keys())), "Class weights must be provided for all classes"

        ### prepare data
        if self.optim_algo in ("sgd", "slbfgs"):
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
            self.w = self._iter_optimize(
                w=self.w,
                dataloader=make_dataloader(),
                direction_fn=self.direction_fn[self.optim_algo],
                use_lbfgs="lbfgs" in self.optim_algo,
                update_callback_kwargs={"epoch": ep, "model": self}
            )
            if self.verbose:
                print(f"[Epoch {ep}/{epochs}] Loss: {self.log_reg_loss(x=x, y=y, w=self.w)}")

        return self

    def predict(self, x, w=None):
        """ Predict the labels of the data

        Parameters:
            x : np.ndarray(N, D) : features
            w : np.ndarray(D) : weights (default to the fitted weights)

        Returns:
            y_pred : np.ndarray(N) : predicted labels
        """
        if self.w is None and w is None:
            raise ValueError("Model has not been trained yet")

        probas = LogisticRegression.sigmoid(x @ self.w if w is None else x @ w)
        y_pred = (probas > self.predict_thres).astype(int)
        return y_pred

    def predict_proba(self, x, w=None):
        """ Predict the probabilities of the data

        Parameters:
            x : np.ndarray(N, D) : features
            w : np.ndarray(D) : weights (default to the fitted weights)

        Returns:
            probas : np.ndarray(N) : predicted probabilities
        """
        if self.w is None and w is None:
            raise ValueError("Model has not been trained yet")

        probas = LogisticRegression.sigmoid(x @ self.w if w is None else x @ w)
        return probas


class SVM:
    """
    Linear Support Vector Machine (SVM) binary classifier

    Parameters:
        _lambda (float) : regularization parameter
        max_iters (int) : maximum number of iterations
        class_weights (dict) : class weights as {class: weight}
    """
    def __init__(self, _lambda=0.1, max_iters=10_000, class_weights={0: 1, 1: 4}):
        self._lambda = _lambda
        self.max_iters = max_iters
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
            self : SVM : fitted SVM instance
        """
        num_examples, num_features = x.shape
        w = np.zeros(num_features)
        alpha = np.zeros(num_examples)

        for it in range(self.max_iters):
            n = np.random.randint(0, num_examples)
            w, alpha = self.calculate_coordinate_update(x, y, alpha, w, n)
        self.w = w

        return self

    def predict(self, x):
        """ Predict the labels for all samples in x
        Parameters:
            x : np.ndarray(N, D) : features
        Returns:
            y_pred : np.ndarray(N) : predicted labels
        """
        return (x @ self.w > 0).astype(int)


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
        self.std_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def _standardize(self, x, use_prev_stats=False):
        """ Standardize the dataset by removing the mean and scaling to unit variance

        Parameters:
            x : np.ndarray(N, D) : data matrix
            use_prev_stats : bool : whether to use previous statistics

        Returns:
            x_standardized : np.ndarray(N, D) : standardized data
        """
        if not use_prev_stats:
            self.mean_ = np.mean(x, axis=0)
            self.std_ = np.std(x, axis=0)
        assert self.mean_ is not None and self.std_ is not None, "Mean and std not computed"
        x_standardized = (x - self.mean_) / (self.std_ + 1e-7)
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
            x = self._standardize(x, use_prev_stats=False)

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

    def transform(self, x, use_prev_stats=True):
        """ Project the data onto the principal components

        Parameters:
            x : np.ndarray(N, D) : data matrix

        Returns:
            x_pca : np.ndarray(N, n_components) : transformed data
        """
        ### standardize the data
        if self.standardize:
            x = self._standardize(x, use_prev_stats=use_prev_stats)

        ### project the data onto the principal components
        return x @ self.components_
