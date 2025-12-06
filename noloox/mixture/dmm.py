"""Implementation from tweetopic: https://github.com/centre-for-humanities-computing/tweetopic/blob/main/tweetopic/dmm.py"""

from __future__ import annotations

import random
import warnings
from typing import Optional

import jax
import numpy as np
import scipy.sparse as spr
from jax.scipy.special import softmax
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClusterMixin, DensityMixin
from sklearn.exceptions import NotFittedError

from noloox.mixture._dmm import fit_model, log_cond_prob


class DirichletMultinomialMixture(BaseEstimator, ClusterMixin, DensityMixin):
    """Implementation of the Dirichlet Multinomial Mixture Model with Gibbs Sampling
    solver

    Parameters
    ----------
    n_components: int
        Number of mixture components in the model.
    n_iter: int, default 50
        Number of iterations during fitting.
        If you find your results are unsatisfactory, increase this number.
    alpha: float, default 0.1
        Willingness of a document joining an empty cluster.
    beta: float, default 0.1
        Willingness to join clusters, where the terms in the document
        are not present.
    random_state: int, default None
        Random seed to use for reproducibility.

    Attributes
    ----------
    components_: array of shape (n_components, n_vocab)
        Describes all components of the topic distribution.
        Contains the amount each word has been assigned to each component
        during fitting.
    n_features_in_: int
        Number of total vocabulary items seen during fitting.
    """

    def __init__(
        self,
        n_components: int,
        n_iter: int = 50,
        alpha: float = 0.1,
        beta: float = 0.1,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state

    def get_params(self, deep: bool = False) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: bool, default False
            Ignored, exists for sklearn compatibility.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        Note
        ----
        Exists for sklearn compatibility.
        """
        return {
            "n_components": self.n_components,
            "n_iter": self.n_iter,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def fit_predict(self, X, y=None):
        """Fits the model using Gibbs Sampling. Detailed description of the
        algorithm in Yin and Wang (2014).

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            BOW matrix of corpus.
        y: None
            Ignored, exists for sklearn compatibility.

        Returns
        -------
        DirichletMultinomialMixture
            The fitted model.

        """
        if issparse(X):
            warnings.warn(
                "Sparse arrays are not yet supported. Implicitly converting to dense array."
            )
            X = np.asarray(X.todense())
        if self.random_state is not None:
            random_key = jax.random.key(self.random_state)
        else:
            random_key = jax.random.key(random.randint(0, 1000))
        random_key, self.components_, self.labels_, self.m_z, self.n_z = fit_model(
            random_key,
            self.n_components,
            self.n_iter,
            self.alpha,
            self.beta,
            X,
        )
        self.weights_ = np.asarray(self.m_z) / np.sum(self.m_z)
        self.components_ = np.asarray(self.components_)
        D, V = X.shape
        self._predict_proba = jax.vmap(
            lambda x: softmax(
                log_cond_prob(
                    self.m_z,
                    self.components_,
                    self.n_z,
                    x,
                    D,
                    self.n_components,
                    V,
                    self.alpha,
                    self.beta,
                )
            ),
        )

        return self.labels_

    def predict_proba(self, X) -> np.ndarray:
        """Predicts probabilities for each document belonging to each
        component.

        Parameters
        ----------
        X: array-like  of shape (n_samples, n_features)
            Document-term matrix.

        Returns
        -------
        array of shape (n_samples, n_components)
            Probabilities for each document belonging to each cluster.

        Raises
        ------
        NotFittedException
            If the model is not fitted, an exception will be raised
        """
        if not hasattr(self, "_predict_proba"):
            raise NotFittedError("Model not fitted yet, can't predict probabilities.")
        if issparse(X):
            warnings.warn(
                "Sparse arrays are not yet supported. Implicitly converting to dense array."
            )
            X = np.asarray(X.todense())
        p = self._predict_proba(X)
        return np.asarray(p)

    def fit(self, X, y=None):
        self.fit_predict(X, y)
        return self

    def transform(self, X) -> np.ndarray:
        """Alias for predict_proba()."""
        return self.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        """Predicts cluster labels for a set of documents. Mainly exists for
        compatibility with density estimators in sklearn.

        Parameters
        ----------
        X: array-like  of shape (n_samples, n_features)
            Document-term matrix.

        Returns
        -------
        array of shape (n_samples,)
            Cluster label for each document.

        Raises
        ------
        NotFittedException
            If the model is not fitted, an exception will be raised
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def fit_transform(
        self,
        X,
        y=None,
    ) -> np.ndarray:
        """Fits the model, then transforms the given data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Document-term matrix.
        y: None
            Ignored, sklearn compatibility.

        Returns
        -------
        array of shape (n_samples, n_components)
            Probabilities for each document belonging to each cluster.
        """
        return self.fit(X).transform(X)
