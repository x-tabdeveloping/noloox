"""This file implements semi-NMF, where doc_topic proportions are not allowed to be negative, but components are unbounded."""

import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import check_is_fitted

EPSILON = np.finfo(np.float32).eps


def init_G(X, n_components: int, constant=0.2, random_state=None) -> np.ndarray:
    """Returns W"""
    kmeans = KMeans(n_components, random_state=random_state).fit(X.T)
    # n_components, n_columns
    G = label_binarize(kmeans.labels_, classes=np.arange(n_components))
    return G + constant


def separate(A):
    abs_A = np.abs(A)
    pos = (abs_A + A) / 2
    neg = (abs_A - A) / 2
    return pos, neg


def update_F(X, G):
    return X @ G @ np.linalg.inv(G.T @ G)


def update_G(X, G, F, l1_reg=0):
    pos_xtf, neg_xtf = separate(X.T @ F)
    pos_gftf, neg_gftf = separate(G @ (F.T @ F))
    numerator = pos_xtf + neg_gftf
    denominator = neg_xtf + pos_gftf
    denominator += l1_reg
    denominator = np.maximum(denominator, EPSILON)
    delta_G = np.sqrt(numerator / denominator)
    G *= delta_G
    return G


def rec_err(X, F, G):
    err = X - (F @ G.T)
    return np.linalg.norm(err)


class SNMF(TransformerMixin, BaseEstimator):
    """Semi-Nonnegative Matrix Factorization.
    This model is quite similar to ordinary Nonnegative matrix factorization,
    but only constrains sources to be nonnegative, `components_` and the outcome variable can still be.

    Parameters
    ----------
    n_components : int
        Number of latent components to estimate.
    tol : float, default=1e-5
        Tolerance for the stopping condition. Training stops when the
        relative reconstruction error improvement falls below this value.
    max_iter : int, default=200
        Maximum number of iterations to perform.
    l1_reg : float, default=0.0
        Strength of L1 regularization applied to the sources.
    random_state : int or None, default=None
        Seed for the random number generator used during initialization.
    verbose : bool, default=False
        If True, prints progress messages during optimization.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Non-constrainted factors.
    n_iter_ : int
        Number of iterations run.
    reconstruction_err_ : float
        Final reconstruction error of the model.


    References
    ----------
    * Ding, C., Li, T., & Jordan, M. I. (2008).
      Convex and Semi-Nonnegative Matrix Factorizations.
      IEEE Transactions on Pattern Analysis and Machine Intelligence.
    """

    def __init__(
        self,
        n_components: int,
        tol: float = 1e-5,
        max_iter: int = 200,
        l1_reg: float = 0.0,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.l1_reg = l1_reg
        self.verbose = verbose

    def fit_transform(self, X: np.ndarray, y=None):
        """Fit the model to `X` and return the transformed data.


        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to be factorized.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        G : ndarray of shape (n_samples, n_components)
            Transformed nonnegative representation of the data.

        Raises
        ------
        UserWarning
            If the algorithm does not converge within `max_iter`
            iterations.
        """
        G = init_G(X.T, self.n_components)
        F = update_F(X.T, G)
        error_at_init = rec_err(X.T, F, G)
        prev_error = error_at_init
        for i in range(
            self.max_iter,
        ):
            G = update_G(X.T, G, F, self.l1_reg)
            F = update_F(X.T, G)
            error = rec_err(X.T, F, G)
            difference = prev_error - error
            if (error < error_at_init) and (
                (prev_error - error) / error_at_init
            ) < self.tol:
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
        self.components_ = F.T
        self.reconstruction_err_ = error
        self.n_iter_ = i
        return np.array(G)

    def transform(self, X: np.ndarray):
        """Transform new data according to the fitted SNMF components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        G : ndarray of shape (n_samples, n_components)
            Nonnegative encoded representation of the input.
        """
        G = np.maximum(X @ np.linalg.pinv(self.components_), 0)
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
