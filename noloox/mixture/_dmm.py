"""Module containing tools for fitting a Dirichlet Multinomial Mixture Model."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

EPS = np.finfo(float).eps

UZERO = jnp.astype(0, "uint32")


def init_clusters(
    cluster_word_distribution: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_doc_count: np.ndarray,
    doc_clusters: np.ndarray,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    max_unique_words: int,
) -> None:
    """Randomly initializes clusters in the model.

    Parameters
    ----------
    cluster_word_count(OUT): array of shape (n_clusters,)
        Contains the amount of words there are in each cluster.
    cluster_word_distribution(OUT): matrix of shape (n_clusters, n_vocab)
        Contains the amount a word occurs in a certain cluster.
    cluster_doc_count(OUT): array of shape (n_clusters,)
        Array containing how many documents there are in each cluster.
    doc_clusters: array of shape (n_docs)
        Contains a cluster label for each document, that has
        to be assigned.
    doc_unique_words: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all indices of unique words in the document.
    doc_unique_word_counts: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all counts for each unique word in the document.
    max_unique_words: int
        Maximum number of unique words in any document.

    NOTE
    ----
    Beware that the function modifies a numpy array, that's passed in as
    an input parameter. Should not be used in parallel, as race conditions
    might arise.
    """
    n_docs, _ = doc_unique_words.shape
    for i_doc in range(n_docs):
        i_cluster = doc_clusters[i_doc]
        cluster_doc_count[i_cluster] += 1
        for i_unique in range(max_unique_words):
            i_word = doc_unique_words[i_doc, i_unique]
            count = doc_unique_word_counts[i_doc, i_unique]
            if not count:
                # Break out when the word is not present in the document
                break
            cluster_word_count[i_cluster] += count
            cluster_word_distribution[i_cluster, i_word] += count


def _cond_prob(
    i_document: int,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    n_words: int,
    alpha: float,
    beta: float,
    n_clusters: int,
    n_vocab: int,
    n_docs: int,
    cluster_doc_count: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_word_distribution: np.ndarray,
    max_unique_words: int,
) -> float:
    """Computes the conditional probability of a certain document joining the
    given mixture component.

    Implements formula no. 4 from Yin & Wang (2014).

    Parameters
    ----------
    i_document: int
        Index of the document in the corpus.
    doc_unique_words: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all indices of unique words in the document.
    doc_unique_word_counts: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all counts for each unique word in the document.
    n_words: int
        Total number of words in the document.
    alpha: float
        Alpha parameter of the model.
    beta: float
        Beta parameter of the model.
    n_clusters: int
        Number of mixture components in the model.
    n_vocab: int
        Number of total vocabulary items.
    n_docs: int
        Total number of documents.
    cluster_doc_count: array of shape (n_clusters,)
        Array containing how many documents there are in each cluster.
    cluster_word_count: array of shape (n_clusters,)
        Contains the amount of words there are in each cluster.
    cluster_word_distribution: matrix of shape (n_clusters, n_vocab)
        Contains the amount a word occurs in a certain cluster.
    max_unique_words: int
        Maximum number of unique words seen in a document.
    """
    log_norm_term = jnp.log(
        (cluster_doc_count + alpha) / (n_docs - 1 + n_clusters * alpha),
    )

    def _numerator_term(doc_unique_words, doc_unique_word_counts):
        i_word = doc_unique_words[i_document]
        count = doc_unique_word_counts[i_document]
        return jax.lax.fori_loop(
            UZERO,
            count,
            lambda j, carry: carry
            + jnp.log(
                cluster_word_distribution[i_word] + beta + j,
            ),
            0,
        )

    numerator_terms = jax.vmap(
        _numerator_term,
    )(doc_unique_words.T, doc_unique_word_counts.T)
    log_numerator = jnp.sum(numerator_terms)
    subres = cluster_word_count + (n_vocab * beta)
    log_denominator = jax.lax.fori_loop(
        UZERO, n_words, lambda j, carry: carry + jnp.log(subres + j), 0
    )
    res = jnp.exp(log_norm_term + log_numerator - log_denominator)
    return res


def predict_doc(
    i_document: int,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    n_words: int,
    alpha: float,
    beta: float,
    n_clusters: int,
    n_vocab: int,
    n_docs: int,
    cluster_doc_count: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_word_distribution: np.ndarray,
    max_unique_words: int,
) -> None:
    """Computes the parameters of the multinomial distribution used for
    sampling.

    Parameters
    ----------
    i_document: int
        Index of the document in the corpus.
    doc_unique_words: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all indices of unique words in the document.
    doc_unique_word_counts: matrix of shape (n_documents, MAX_UNIQUE_WORDS)
        Matrix containing all counts for each unique word in the document.
    n_words: int
        Total number of words in the document.
    alpha: float
        Alpha parameter of the model.
    beta: float
        Beta parameter of the model.
    n_clusters: int
        Number of mixture components in the model.
    n_vocab: int
        Number of total vocabulary items.
    n_docs: int
        Total number of documents.
    cluster_doc_count: array of shape (n_clusters,)
        Array containing how many documents there are in each cluster.
    cluster_word_count: array of shape (n_clusters,)
        Contains the amount of words there are in each cluster.
    cluster_word_distribution: matrix of shape (n_clusters, n_vocab)
        Contains the amount a word occurs in a certain cluster.
    max_unique_words: int
        Maximum number of unique words seen in a document.
    """

    cond_prob = jax.vmap(
        lambda cluster_doc_count, cluster_word_count, cluster_word_distribution: _cond_prob(
            i_document=i_document,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            n_words=n_words,
            alpha=alpha,
            beta=beta,
            n_clusters=n_clusters,
            n_vocab=n_vocab,
            n_docs=n_docs,
            cluster_doc_count=cluster_doc_count,
            cluster_word_count=cluster_word_count,
            cluster_word_distribution=cluster_word_distribution,
            max_unique_words=max_unique_words,
        ),
        in_axes=(0, 0, 0),
        out_axes=0,
    )
    probabilities = cond_prob(
        cluster_doc_count, cluster_word_count, cluster_word_distribution
    )
    probabilities = probabilities / jnp.maximum(jnp.sum(probabilities), EPS)
    return probabilities


def _remove_one(
    cluster_word_distribution,
    cluster_word_count,
    doc_unique_words,
    doc_unique_word_counts,
    i_cluster,
    i_doc,
    i_unique,
):
    i_word = doc_unique_words[i_doc, i_unique]
    count = doc_unique_word_counts[i_doc, i_unique]
    cluster_word_count = cluster_word_count.at[i_cluster].set(
        cluster_word_count[i_cluster] - count
    )
    cluster_word_distribution = cluster_word_distribution.at[i_cluster, i_word].set(
        cluster_word_distribution[i_cluster, i_word] - count
    )
    return cluster_word_distribution, cluster_word_count


def _add_one(
    cluster_word_distribution,
    cluster_word_count,
    doc_unique_words,
    doc_unique_word_counts,
    i_cluster,
    i_doc,
    i_unique,
):
    i_word = doc_unique_words[i_doc, i_unique]
    count = doc_unique_word_counts[i_doc, i_unique]
    cluster_word_count = cluster_word_count.at[i_cluster].set(
        cluster_word_count[i_cluster] + count
    )
    cluster_word_distribution = cluster_word_distribution.at[i_cluster, i_word].set(
        cluster_word_distribution[i_cluster, i_word] + count
    )
    return cluster_word_distribution, cluster_word_count


def remove_doc(
    cluster_doc_count,
    cluster_word_distribution,
    cluster_word_count,
    doc_unique_words,
    doc_unique_word_counts,
    max_unique_words,
    i_cluster,
    i_doc,
):
    cluster_doc_count = cluster_doc_count.at[i_cluster].set(
        cluster_doc_count[i_cluster] - 1
    )
    cluster_word_distribution, cluster_word_count = jax.lax.fori_loop(
        0,
        max_unique_words,
        lambda i_unique, params: _remove_one(
            params[0],
            params[1],
            doc_unique_words,
            doc_unique_word_counts,
            i_cluster,
            i_doc,
            i_unique,
        ),
        (cluster_word_distribution, cluster_word_count),
    )
    return cluster_doc_count, cluster_word_distribution, cluster_word_count


def add_doc(
    cluster_doc_count,
    cluster_word_distribution,
    cluster_word_count,
    doc_unique_words,
    doc_unique_word_counts,
    max_unique_words,
    i_cluster,
    i_doc,
):
    cluster_doc_count = cluster_doc_count.at[i_cluster].set(
        cluster_doc_count[i_cluster] + 1
    )
    cluster_word_distribution, cluster_word_count = jax.lax.fori_loop(
        0,
        max_unique_words,
        lambda i_unique, params: _add_one(
            params[0],
            params[1],
            doc_unique_words,
            doc_unique_word_counts,
            i_cluster,
            i_doc,
            i_unique,
        ),
        (cluster_word_distribution, cluster_word_count),
    )
    return cluster_doc_count, cluster_word_distribution, cluster_word_count


def _doc_step(
    i_doc,
    random_key,
    alpha,
    beta,
    n_clusters,
    n_vocab,
    n_docs,
    doc_unique_words,
    doc_unique_word_counts,
    doc_clusters,
    cluster_doc_count,
    cluster_word_count,
    cluster_word_distribution,
    max_unique_words,
    doc_word_count,
):
    # Removing document from previous cluster
    prev_cluster = doc_clusters[i_doc]
    cluster_doc_count, cluster_word_distribution, cluster_word_count = remove_doc(
        cluster_doc_count,
        cluster_word_distribution,
        cluster_word_count,
        doc_unique_words,
        doc_unique_word_counts,
        max_unique_words,
        prev_cluster,
        i_doc,
    )
    # Getting new prediction for the document at hand
    prediction = predict_doc(
        i_document=i_doc,
        doc_unique_words=doc_unique_words,
        doc_unique_word_counts=doc_unique_word_counts,
        n_words=doc_word_count[i_doc],
        alpha=alpha,
        beta=beta,
        n_clusters=n_clusters,
        n_vocab=n_vocab,
        n_docs=n_docs,
        cluster_doc_count=cluster_doc_count,
        cluster_word_count=cluster_word_count,
        cluster_word_distribution=cluster_word_distribution,
        max_unique_words=max_unique_words,
    )
    key, random_key = jax.random.split(random_key)
    logits = jnp.log(prediction / (1 - prediction))
    new_cluster = jax.random.categorical(key, logits)
    # Adding document back to the newly chosen cluster
    cluster_doc_count, cluster_word_distribution, cluster_word_count = add_doc(
        cluster_doc_count,
        cluster_word_distribution,
        cluster_word_count,
        doc_unique_words,
        doc_unique_word_counts,
        max_unique_words,
        new_cluster,
        i_doc,
    )
    doc_clusters = doc_clusters.at[i_doc].set(new_cluster)
    return (
        random_key,
        doc_clusters,
        cluster_doc_count,
        cluster_word_distribution,
        cluster_word_count,
    )


@jax.jit
def _sampling_step(
    random_key,
    alpha: float,
    beta: float,
    n_clusters: int,
    n_vocab: int,
    n_docs: int,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    doc_clusters: np.ndarray,
    cluster_doc_count: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_word_distribution: np.ndarray,
    max_unique_words: int,
    prediction: np.ndarray,
    doc_word_count: np.ndarray,
) -> None:
    return jax.lax.fori_loop(
        0,
        n_docs,
        lambda i_doc, params: _doc_step(
            i_doc,
            params[0],
            alpha,
            beta,
            n_clusters,
            n_vocab,
            n_docs,
            doc_unique_words,
            doc_unique_word_counts,
            params[1],
            params[2],
            params[4],
            params[3],
            max_unique_words,
            doc_word_count,
        ),
        (
            random_key,
            doc_clusters,
            cluster_doc_count,
            cluster_word_distribution,
            cluster_word_count,
        ),
    )


def fit_model(
    random_key,
    n_iter: int,
    alpha: float,
    beta: float,
    n_clusters: int,
    n_vocab: int,
    n_docs: int,
    doc_unique_words: np.ndarray,
    doc_unique_word_counts: np.ndarray,
    doc_clusters: np.ndarray,
    cluster_doc_count: np.ndarray,
    cluster_word_count: np.ndarray,
    cluster_word_distribution: np.ndarray,
    max_unique_words: int,
) -> None:
    doc_word_count = np.sum(doc_unique_word_counts, axis=1)
    prediction = np.empty(n_clusters)
    for _ in trange(n_iter, desc="Sampling from posterior"):
        (
            random_key,
            doc_clusters,
            cluster_doc_count,
            cluster_word_distribution,
            cluster_word_count,
        ) = _sampling_step(
            random_key,
            alpha=alpha,
            beta=beta,
            n_clusters=n_clusters,
            n_vocab=n_vocab,
            n_docs=n_docs,
            doc_unique_words=doc_unique_words,
            doc_unique_word_counts=doc_unique_word_counts,
            doc_clusters=doc_clusters,
            cluster_doc_count=cluster_doc_count,
            cluster_word_count=cluster_word_count,
            cluster_word_distribution=cluster_word_distribution,
            max_unique_words=max_unique_words,
            doc_word_count=doc_word_count,
            prediction=prediction,
        )
    return (
        random_key,
        doc_clusters,
        cluster_doc_count,
        cluster_word_distribution,
        cluster_word_count,
    )
