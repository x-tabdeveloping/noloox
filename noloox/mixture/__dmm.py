from functools import partial

import jax
import jax.numpy as jnp


def _numerator_term(dt, n, beta):
    return jax.lax.fori_loop(
        0,
        dt,
        lambda j, carry: carry + jnp.log(j + beta + n),
    )


def _cond_prob(m, n_w, n_d, doc_term, Nd, D, K, V, alpha, beta):
    log_norm_term = jnp.log(
        (m + alpha) / (D - 1 + K * alpha),
    )
    log_numerator = jax.sum(
        jax.vmap(partial(_numerator_term, beta=beta), (0, 0), 0)(doc_term, n_w)
    )
    log_denominator = jax.lax.fori_loop(
        0, Nd, lambda j, carry: carry + jnp.log(n_d + V * beta + j - 1)
    )
    return log_norm_term + log_numerator - log_denominator


def log_cond_prob(m_z, n_z_w, n_z_d, doc_term, Nd, D, K, V, alpha, beta):
    return jax.vmap(
        partial(
            _cond_prob, doc_term=doc_term, Nd=Nd, D=D, K=K, V=V, alpha=alpha, beta=beta
        ),
        (0, 0, 0),
        0,
    )(m_z, n_z_w, n_z_d)


def fit_model(
    random_key,
    n_iter: int,
    alpha: float,
    beta: float,
    doc_term_matrix,
    doc_topic_matrix,
    components_,
):
    for _ in trange(n_iter, desc="Sampling from posterior"):
        for i_doc in range(len(doc_topic_matrix)):
            # Removing document from previous cluster
            current_cluster = jnp.argmax(doc_topic_matrix[i_doc])
            doc_topic_matrix[i_doc, current_cluster] = 0
            components_[current_cluster] -= doc_term_matrix[i_doc]
            # Getting new prediction for the document at hand
            log_p = log_cond_prob()
            random_key, subkey = jax.random.split(random_key)
            new_cluster = jnp.argmax(jax.random.multinomial(subkey, n=1, p=prediction))
            # Adding document back to the newly chosen cluster
            components_[new_cluster] += doc_term_matrix[i_doc]
            doc_topic_matrix[i_doc, new_cluster] = 1
    return doc_topic_matrix, components_
