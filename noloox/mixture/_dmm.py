from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import softmax
from tqdm import trange


def _numerator_term(dt, n, beta):
    return jax.lax.fori_loop(0, dt, lambda j, carry: carry + jnp.log(j + beta + n), 0)


def _cond_prob(m, n_w, n, doc_term, Nd, D, K, V, alpha, beta):
    log_norm_term = jnp.log(
        (m + alpha) / (D - 1 + K * alpha),
    )
    log_numerator = jnp.sum(
        jax.vmap(partial(_numerator_term, beta=beta), (0, 0), 0)(doc_term, n_w)
    )
    log_denominator = jax.lax.fori_loop(
        0, Nd, lambda j, carry: carry + jnp.log(n + V * beta + j - 1), 0
    )
    return log_norm_term + log_numerator - log_denominator


def log_cond_prob(m_z, n_z_w, n_z, doc_term, D, K, V, alpha, beta):
    Nd = jnp.sum(doc_term)
    return jax.vmap(
        partial(
            _cond_prob, doc_term=doc_term, Nd=Nd, D=D, K=K, V=V, alpha=alpha, beta=beta
        ),
        (0, 0, 0),
        0,
    )(m_z, n_z_w, n_z)


def _doc_step(random_key, components, m_z, n_z, doc_term, doc_z, D, K, V, alpha, beta):
    S = doc_term.sum()
    # Removing document from previous cluster
    components = components.at[doc_z].subtract(doc_term)
    m_z = m_z.at[doc_z].subtract(1)
    n_z = n_z.at[doc_z].subtract(S)
    # Getting new prediction for the document at hand
    log_p = log_cond_prob(m_z, components, n_z, doc_term, D, K, V, alpha, beta)
    random_key, subkey = jax.random.split(random_key)
    z = jnp.argmax(jax.random.multinomial(subkey, n=1, p=softmax(log_p)))
    # Adding document back to the newly chosen cluster
    components = components.at[z].add(doc_term)
    m_z = m_z.at[z].add(1)
    n_z = n_z.at[z].add(S)
    return (random_key, components, m_z, n_z), z


def _step(
    components, random_key, doc_zs, m_z, n_z, doc_term_matrix, D, K, V, alpha, beta
):
    def doc_step(carry, xs):
        random_key, components, m_z, n_z = carry
        z, doc_term = xs
        return _doc_step(
            random_key,
            components,
            m_z,
            n_z,
            doc_term,
            doc_z=z,
            D=D,
            K=K,
            V=V,
            alpha=alpha,
            beta=beta,
        )

    (random_key, components, m_z, n_z), zs = jax.lax.scan(
        doc_step, init=(random_key, components, m_z, n_z), xs=(doc_zs, doc_term_matrix)
    )
    return random_key, components, zs, m_z, n_z


def init_model(
    n_components: int,
    doc_term_matrix,
    random_key,
) -> None:
    """Randomly initializes clusters in the model."""
    D, V = doc_term_matrix.shape
    m_z = np.zeros(n_components)
    n_z = np.zeros(n_components)
    components = np.zeros((n_components, V))
    random_key, subkey = jax.random.split(random_key)
    zs = jnp.argmax(
        jax.random.multinomial(
            subkey, n=1, p=np.ones(n_components) / n_components, shape=(D, n_components)
        ),
        axis=1,
    )
    for z, doc_term in zip(zs, doc_term_matrix):
        m_z[z] += 1
        n_z[z] += doc_term.sum()
        components[z] += np.ravel(np.asarray(doc_term))
    return random_key, components, zs, m_z, n_z


def fit_model(
    random_key,
    n_components,
    n_iter,
    alpha,
    beta,
    doc_term_matrix,
):
    D, V = doc_term_matrix.shape
    K = n_components
    random_key, components, zs, m_z, n_z = init_model(
        n_components, doc_term_matrix, random_key
    )

    @jax.jit
    def step(random_key, components, zs, m_z, n_z):
        return _step(
            components,
            random_key,
            zs,
            m_z,
            n_z,
            doc_term_matrix,
            D,
            K,
            V,
            alpha,
            beta,
        )

    for _ in trange(n_iter, desc="Sampling from posterior"):
        random_key, components, zs, m_z, n_z = step(
            random_key, components, zs, m_z, n_z
        )
    return random_key, components, zs, m_z, n_z
