"""Implementation from tweetopic: https://github.com/centre-for-humanities-computing/tweetopic/blob/main/tweetopic/dmm.py"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as spr
from sklearn.base import BaseEstimator, ClusterMixin, DensityMixin

from noloox.mixture._dmm import fit_model, init_clusters, predict_doc


def init_doc_words(
    doc_term_matrix: spr.lil_matrix,
    max_unique_words: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_docs, _ = doc_term_matrix.shape
    doc_unique_words = np.zeros((n_docs, max_unique_words)).astype(np.uint32)
    doc_unique_word_counts = np.zeros((n_docs, max_unique_words)).astype(np.uint32)
    for i_doc in range(n_docs):
        unique_words = doc_term_matrix[i_doc].rows[0]  # type: ignore
        unique_word_counts = doc_term_matrix[i_doc].data[0]  # type: ignore
        for i_unique in range(len(unique_words)):
            doc_unique_words[i_doc, i_unique] = unique_words[i_unique]
            doc_unique_word_counts[i_doc, i_unique] = unique_word_counts[i_unique]
    return doc_unique_words, doc_unique_word_counts


class DirichletMultinomialMixture(BaseEstimator, ClusterMixin, DensityMixin):
    """Implementation of the Dirichlet Multinomial Mixture Model with Gibbs Sampling
    solver

    Parameters
    ----------
    n_components: int
        Number of mixture components in the model.
    n_iterations: int, default 50
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
        n_iterations: int = 50,
        alpha: float = 0.1,
        beta: float = 0.1,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_components = n_components
        self.n_iterations = n_iterations
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
            "n_iterations": self.n_iterations,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def fit(self, X, y=None):
        """Fits the model using Gibbs Sampling. Detailed description of the
        algorithm in Yin and Wang (2014).

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_samples, n_features)
            BOW matrix of corpus.
        y: None
            Ignored, exists for sklearn compatibility.

        Returns
        -------
        DirichletMultinomialMixture
            The fitted model.

        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # Converting X into sparse array if it isn't one already.
        X = spr.csr_matrix(X)
        self.n_documents, self.n_features_in_ = X.shape
        # Calculating the number of nonzero elements for each row
        # using the internal properties of CSR matrices.
        self.max_unique_words = np.max(np.diff(X.indptr))
        doc_unique_words, doc_unique_word_counts = init_doc_words(
            X.tolil(),
            max_unique_words=self.max_unique_words,
        )
        initial_clusters = np.random.multinomial(
            1,
            np.ones(self.n_components) / self.n_components,
            size=self.n_documents,
        )
        doc_clusters = np.argmax(initial_clusters, axis=1)
        self.cluster_doc_count = np.zeros(self.n_components)
        self.components_ = np.zeros((self.n_components, self.n_features_in_))
        self.cluster_word_count = np.zeros(self.n_components)
        init_clusters(
            cluster_word_distribution=self.components_,
            cluster_word_count=self.cluster_word_count,
            cluster_doc_count=self.cluster_doc_count,
            doc_clusters=doc_clusters,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            max_unique_words=self.max_unique_words,
        )
        fit_model(
            n_iter=self.n_iterations,
            alpha=self.alpha,
            beta=self.beta,
            n_clusters=self.n_components,
            n_vocab=self.n_features_in_,
            n_docs=self.n_documents,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            doc_clusters=doc_clusters,
            cluster_doc_count=self.cluster_doc_count,
            cluster_word_count=self.cluster_word_count,
            cluster_word_distribution=self.components_,
            max_unique_words=self.max_unique_words,
        )
        self.weights_ = self.cluster_doc_count / np.sum(self.cluster_doc_count)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predicts probabilities for each document belonging to each
        component.

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_samples, n_features)
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
        # Converting X into sparse array if it isn't one already.
        X = spr.csr_matrix(X)
        sample_max_unique_words = np.max(np.diff(X.indptr))
        doc_unique_words, doc_unique_word_counts = init_doc_words(
            X.tolil(),
            max_unique_words=sample_max_unique_words,
        )
        doc_words_count = np.sum(doc_unique_word_counts, axis=1)
        n_docs = X.shape[0]
        predictions = []
        for i_doc in range(n_docs):
            pred = np.zeros(self.n_components)
            predict_doc(
                probabilities=pred,
                i_document=i_doc,
                doc_unique_words=doc_unique_words,
                doc_unique_word_counts=doc_unique_word_counts,
                n_words=doc_words_count[i_doc],
                alpha=self.alpha,
                beta=self.beta,
                n_clusters=self.n_components,
                n_vocab=self.n_features_in_,
                n_docs=n_docs,
                cluster_doc_count=self.cluster_doc_count,
                cluster_word_count=self.cluster_word_count,
                cluster_word_distribution=self.components_,
                max_unique_words=sample_max_unique_words,
            )
            predictions.append(pred)
        return np.stack(predictions)

    def transform(self, X) -> np.ndarray:
        """Alias for predict_proba()."""
        return self.transform(X)

    def predict(self, X) -> np.ndarray:
        """Predicts cluster labels for a set of documents. Mainly exists for
        compatibility with density estimators in sklearn.

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_samples, n_features)
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
        X: array-like or sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        y: None
            Ignored, sklearn compatibility.

        Returns
        -------
        array of shape (n_samples, n_components)
            Probabilities for each document belonging to each cluster.
        """
        return self.fit(X).transform(X)

    def fit_predict(
        self,
        X,
        y=None,
    ) -> np.ndarray:
        """Fits the model, then transforms the given data.

        Parameters
        ----------
        X: array-like or sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        y: None
            Ignored, sklearn compatibility.

        Returns
        -------
        array of shape (n_samples, n_components)
            Probabilities for each document belonging to each cluster.
        """
        return self.fit(X).predict(X)
