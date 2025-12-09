from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition._nmf import _beta_divergence, _initialize_nmf
from sklearn.utils.extmath import safe_sparse_dot

EPSILON = np.finfo(np.float32).eps


class WNMF(TransformerMixin, BaseEstimator):
    """
    Weighted Nonnegative Matrix Factorization (WNMF).

    WNMF factorizes a nonnegative data matrix $X$ into two
    nonnegative matrices $W$ and $H$ such that

    $X \\approx W H$

    but introduces per-entry weights given by `y`.
    The weights modify the update rules so that reconstruction errors on
    entries with higher weights contribute more strongly to the objective.

    The optimization uses multiplicative updates for both factors
    :math:`U` and :math:`V` following a weighted Euclidean (β=2)
    divergence objective.

    Parameters
    ----------
    n_components : int
        Number of latent components to use.
    max_iter : int, default=200
        Maximum number of iterations before stopping.
    tol : float, default=1e-4
        Tolerance for early stopping. The algorithm checks every 10 iterations
        whether the reconstruction error has ceased improving.
    random_state : int or None, default=None
        Seed for random initialization of the factors.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The learned basis matrix :math:`U^T`.

    References
    ----------
    Y. -D. Kim and S. Choi, "Weighted nonnegative matrix factorization," 2009 IEEE International Conference on Acoustics, Speech and Signal Processing, Taipei, Taiwan, 2009, pp. 1541-1544, doi: 10.1109/ICASSP.2009.4959890. keywords: {Matrix decomposition;Least squares approximation;Collaboration;Convergence;Least squares methods;Data analysis;Computer science;Singular value decomposition;Feature extraction;Spectrogram;Alternating nonnegative least squares;collaborative prediction;generalized EM;nonnegative matrix factorization;weighted low-rank approximation},
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        """Fit the WNMF model to data `X` with weights `y` and return the
        transformed representation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Nonnegative data matrix to factorize.

        y : ndarray of shape (n_samples,)
            Weights applied to each entry of `X`.
            Larger values increase the importance of corresponding entries.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            The learned encoding matrix $V^T$ of the data.
        """
        X_transformed, components = _initialize_nmf(
            X, self.n_components, random_state=self.random_state
        )
        U = components.T
        V = X_transformed.T
        weighted_A = X.T * y  # .T
        prev_error = np.inf
        for i in range(0, self.max_iter):
            # Update V
            numerator = safe_sparse_dot(U.T, weighted_A)
            denominator = np.linalg.multi_dot((U.T, U, V * y))
            denominator[denominator <= 0] = EPSILON
            delta = numerator
            delta /= denominator
            delta[np.isinf(delta) & (V == 0)] = 0
            V *= delta
            # Update U
            numerator = safe_sparse_dot(weighted_A, V.T)
            denominator = np.linalg.multi_dot((U, V * y, V.T))
            denominator[denominator <= 0] = EPSILON
            delta = numerator
            delta /= denominator
            delta[np.isinf(delta) & (U == 0)] = 0
            U *= delta
            if (self.tol > 0) and (i % 10 == 0):
                error = _beta_divergence(X, V.T, U.T, 2)
                if (error - prev_error) > self.tol:
                    break
                prev_error = error
        self.components_ = U.T
        X_transformed = V.T
        return X_transformed

    def fit(self, X, y=None):
        """Fit the WNMF model to data `X` with weights `y`.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Nonnegative data matrix to factorize.

        y : ndarray of shape (n_samples,)
            Weights applied to each entry of `X`.
            Larger values increase the importance of corresponding entries.

        Returns
        -------
        self
            Fitted WNMF model.
        """
        self.fit_transform(X, y)
        return self

    def transform(self, X: np.ndarray):
        """Transform new data according to the fitted WNMF components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Nonnegative encoded representation of the input.
        """
        X_transformed = np.maximum(X @ np.linalg.pinv(self.components_), 0)
        return np.array(X_transformed)

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
