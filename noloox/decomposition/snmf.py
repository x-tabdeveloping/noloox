"""This file implements semi-NMF, where doc_topic proportions are not allowed to be negative, but components are unbounded."""

import warnings
from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import jit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from tqdm import trange

EPSILON = np.finfo(np.float32).eps


def init_G(X, n_components: int, constant=0.2, random_state=None) -> np.ndarray:
    """Returns W"""
    kmeans = KMeans(n_components, random_state=random_state).fit(X.T)
    # n_components, n_columns
    G = label_binarize(kmeans.labels_, classes=np.arange(n_components))
    return G + constant


def separate(A):
    abs_A = jnp.abs(A)
    pos = (abs_A + A) / 2
    neg = (abs_A - A) / 2
    return pos, neg


def update_F(X, G):
    return X @ G @ jnp.linalg.inv(G.T @ G)


def update_G(X, G, F, sparsity=0):
    pos_xtf, neg_xtf = separate(X.T @ F)
    pos_gftf, neg_gftf = separate(G @ (F.T @ F))
    numerator = pos_xtf + neg_gftf
    denominator = neg_xtf + pos_gftf
    denominator += sparsity
    denominator = jnp.maximum(denominator, EPSILON)
    delta_G = jnp.sqrt(numerator / denominator)
    G *= delta_G
    return G


def rec_err(X, F, G):
    err = X - (F @ G.T)
    return jnp.linalg.norm(err)


@jit
def step(G, F, X, sparsity=0):
    G = update_G(X.T, G, F, sparsity)
    F = update_F(X.T, G)
    error = rec_err(X.T, F, G)
    return G, F, error


class SNMF(TransformerMixin, BaseEstimator):
    """Semi-Nonnegative Matrix Factorization.
    Equivalent to NMF, except the components, and therefore the outcome variables are unbounded.
    The latent factors are constrained to be nonnegative.

    Example:
    ```python
    import numpy as np
    from noloox.decomposition import SNMF

    X = np.random.normal(0, 1, size=(200, 50))
    model = SNMF(n_components=10)

    X_transformed = model.fit_transform(X)
    assert np.all(X_transformed >= 0)
    ```

    Parameters
    ----------
    n_components: int
        Number of latent components to discover.
    tol: float, default=1e-5
        Tolerance for stopping condition.
    max_iter: int, default=200
        Maximum number of iterations.
    progress_bar: bool, default=True
        Indicates whether to display a progress bar when fitting.
    random_state: int, default=None
        Used for model intialization with KMeans.
    sparsity: float, default=0.0
        L1 penalty. Higher values result in a stricter clustering.

    Attributes
    ----------
    components_: ndarray of shape (n_components, n_features)
        Factorization matrix, sometimes called ‘dictionary’.
        Unconstrained.
    n_iter_: int
        Acutal number of iterations.
    reconstruction_err_: float
        Reconstruction error of the model at the last iteration.
    """

    def __init__(
        self,
        n_components: int,
        tol: float = 1e-5,
        max_iter: int = 200,
        progress_bar: bool = True,
        random_state: Optional[int] = None,
        sparsity: float = 0.0,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.progress_bar = progress_bar
        self.random_state = random_state
        self.sparsity = sparsity
        self.verbose = verbose

    def fit_transform(self, X, y=None):
        """Learn an SNMF model for the data X and returns the transformed data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Datapoints to factor.
        y: Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data. Strictily nonnegative.
        """
        G = init_G(X.T, self.n_components, random_state=self.random_state)
        F = update_F(X.T, G)
        error_at_init = rec_err(X.T, F, G)
        prev_error = error_at_init
        _step = partial(step, sparsity=self.sparsity, X=X)
        for i in trange(
            self.max_iter,
            desc="Iterative updates.",
            disable=not self.progress_bar,
        ):
            G, F, error = _step(G, F)
            difference = prev_error - error
            if (error < error_at_init) and (
                (prev_error - error) / error_at_init
            ) < self.tol:
                if self.verbose:
                    print(f"Converged after {i} iterations")
                self.n_iter_ = i
                break
            prev_error = error
            if self.verbose:
                print(
                    f"Iteration: {i}, Error: {error}, init_error: {error_at_init}, difference from previous: {difference}"
                )
        else:
            warnings.warn("SNMF did not converge, try specifying a higher max_iter.")
        self.components_ = np.array(F.T)
        self.reconstruction_err_ = error
        self.n_iter_ = i
        return np.array(G)

    def fit(self, X, y=None):
        """Learn an SNMF model for the data X.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Datapoints to factor.
        y: Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self: SNMF
            Fitted model.
        """
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        """Transform the data X according to the fitted SNMF model.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Datapoints to transform.

        Returns
        -------
        W: ndarray of shape (n_samples, n_components)
            Nonnegative latent sources.
        """
        G = jnp.maximum(X @ jnp.linalg.pinv(self.components_), 0)
        return np.array(G)

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_components)
            Transformed data matrix.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Returns a data matrix of the original shape.
        """
        return X @ self.components_
