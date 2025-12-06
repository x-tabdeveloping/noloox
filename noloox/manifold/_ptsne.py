import os

import jax
import jax.numpy as jnp
import keras.backend as K
import numpy as np
import sklearn
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, InputLayer
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

EPSILON = np.finfo(np.float32).eps


def Hbeta(D, beta):
    P = jnp.exp(-D * beta)
    sumP = jnp.maximum(jnp.sum(P), EPSILON)
    H = jnp.log(sumP) + beta * jnp.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p_job(data, max_iteration=50, tol=1e-5):
    i, Di, logU = data
    beta = 1.0
    beta_min = -jnp.inf
    beta_max = jnp.inf
    H, thisP = Hbeta(Di, beta)
    Hdiff = H - logU

    def cond(state):
        _, _, _, Hdiff, tries, _ = state
        return jnp.logical_and(tries < max_iteration, jnp.abs(Hdiff) > tol)

    def loop_body(state):
        beta_min, beta_max, beta, Hdiff, tries, _ = state
        beta_min, beta_max, beta = jax.lax.cond(
            Hdiff > 0,
            lambda: (
                beta,
                beta_max,
                jax.lax.cond(
                    jnp.isposinf(beta_max),
                    lambda: beta * 2,
                    lambda: (beta + beta_max) / 2.0,
                ),
            ),
            lambda: (
                beta_min,
                beta,
                jax.lax.cond(
                    jnp.isneginf(beta_min),
                    lambda: beta / 2,
                    lambda: (beta + beta_min) / 2.0,
                ),
            ),
        )
        H, thisP = Hbeta(Di, beta)
        return beta_min, beta_max, beta, H - logU, tries + 1, thisP

    state = (beta_min, beta_max, beta, Hdiff, 0, thisP)
    beta_min, beta_max, beta, Hdiff, tries, thisP = jax.lax.while_loop(
        cond, loop_body, init_val=state
    )
    return i, thisP


def x2p(X, perplexity, n_jobs=None):
    n = X.shape[0]
    logU = np.log(perplexity)
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X + (sum_X.reshape((-1, 1)) - 2 * np.dot(X, X.T))
    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape((n, -1))
    P = np.zeros([n, n])
    for i in range(n):
        P[i, idx[i]] = x2p_job((i, D[i], logU))[1]
    return P


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


class ParametricTSNE(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        n_iter=1000,
        batch_size=500,
        early_exaggeration_epochs=50,
        early_exaggeration_value=4.0,
        early_stopping_epochs=np.inf,
        early_stopping_min_improvement=1e-2,
        alpha=1,
        nl1=1000,
        nl2=500,
        nl3=250,
        logdir=None,
        verbose=0,
    ):

        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.verbose = verbose

        # FFNet architecture
        self.nl1 = nl1
        self.nl2 = nl2
        self.nl3 = nl3

        # Early-exaggeration
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.early_exaggeration_value = early_exaggeration_value
        # Early-stopping
        self.early_stopping_epochs = early_stopping_epochs
        self.early_stopping_min_improvement = early_stopping_min_improvement

        # t-Student params
        self.alpha = alpha

        # Tensorboard
        self.logdir = logdir

        # Internals
        self._model = None

    def fit(self, X, y=None):
        """fit the model with X"""

        if self.batch_size is None:
            self.batch_size = X.shape[0]
        else:
            # HACK! REDUCE 'X' TO MAKE IT MULTIPLE OF BATCH_SIZE!
            m = X.shape[0] % self.batch_size
            if m > 0:
                X = X[:-m]

        n_sample, n_feature = X.shape

        self._log("Building model..", end=" ")
        self._build_model(n_feature, self.n_components)
        self._log("Done")

        self._log("Start training..")

        # Tensorboard
        if not self.logdir == None:
            callback = TensorBoard(self.logdir)
            callback.set_model(self._model)
        else:
            callback = None

        # Early stopping
        es_patience = self.early_stopping_epochs
        es_loss = np.inf
        es_stop = False

        # Precompute P (once for all!)
        P = self._calculate_P(X)

        epoch = 0
        while epoch < self.n_iter and not es_stop:

            # Make copy
            _P = P.copy()

            ## Shuffle entries
            # p_idxs = np.random.permutation(self.batch_size)

            # Early exaggeration
            if epoch < self.early_exaggeration_epochs:
                _P *= self.early_exaggeration_value

            # Actual training
            loss = 0.0
            n_batches = 0
            for i in range(0, n_sample, self.batch_size):

                batch_slice = slice(i, i + self.batch_size)
                X_batch, _P_batch = X[batch_slice], _P[batch_slice]

                # Shuffle entries
                p_idxs = np.random.permutation(self.batch_size)
                # Shuffle data
                X_batch = X_batch[p_idxs]
                # Shuffle rows and cols of P
                _P_batch = _P_batch[p_idxs, :]
                _P_batch = _P_batch[:, p_idxs]

                loss += self._model.train_on_batch(X_batch, _P_batch)
                n_batches += 1

            # End-of-epoch: summarize
            loss /= n_batches

            if epoch % 10 == 0:
                self._log("Epoch: {0} - Loss: {1:.3f}".format(epoch, loss))

            if callback is not None:
                # Write log
                write_log(callback, ["loss"], [loss], epoch)

            # Check early-stopping condition
            if (
                loss < es_loss
                and np.abs(loss - es_loss) > self.early_stopping_min_improvement
            ):
                es_loss = loss
                es_patience = self.early_stopping_epochs
            else:
                es_patience -= 1

            if es_patience == 0:
                self._log("Early stopping!")
                es_stop = True

            # Going to the next iteration...
            del _P
            epoch += 1

        self._log("Done")

        return self  # scikit-learn does so..

    def transform(self, X):
        """apply dimensionality reduction to X"""
        # fit should have been called before
        if self.model is None:
            raise sklearn.exceptions.NotFittedError(
                "This ParametricTSNE instance is not fitted yet. Call 'fit'"
                " with appropriate arguments before using this method."
            )

        self._log("Predicting embedding points..", end=" ")
        X_new = self.model.predict(X, batch_size=X.shape[0])

        self._log("Done")

        return X_new

    def fit_transform(self, X, y=None):
        """fit the model with X and apply the dimensionality reduction on X."""
        self.fit(X, y)

        X_new = self.transform(X)
        return X_new

    # ================================ Internals ================================

    def _calculate_P(self, X):
        n = X.shape[0]
        P = np.zeros([n, self.batch_size])
        self._log("Computing P...")
        for i in tqdm(np.arange(0, n, self.batch_size)):
            P_batch = x2p(X[i : i + self.batch_size], self.perplexity)
            P_batch[np.isnan(P_batch)] = 0
            P_batch = P_batch + P_batch.T
            P_batch = P_batch / P_batch.sum()
            P_batch = np.maximum(P_batch, 1e-12)
            P[i : i + self.batch_size] = P_batch
        return P

    def _kl_divergence(self, P, Y):
        sum_Y = K.sum(K.square(Y), axis=1)
        eps = K.variable(1e-15)
        D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
        Q = K.pow(1 + D / self.alpha, -(self.alpha + 1) / 2)
        Q *= K.variable(1 - np.eye(self.batch_size))
        Q /= K.sum(Q)
        Q = K.maximum(Q, eps)
        C = K.log((P + eps) / (Q + eps))
        C = K.sum(P * C)

        return C

    def _build_model(self, n_input, n_output):
        self._model = Sequential()
        self._model.add(InputLayer((n_input,)))
        # Layer adding loop
        for n in [self.nl1, self.nl2, self.nl3]:
            self._model.add(Dense(n, activation="relu"))
        self._model.add(Dense(n_output, activation="linear"))
        self._model.compile("adam", self._kl_divergence)

    def _log(self, *args, **kwargs):
        """logging with given arguments and keyword arguments"""
        if self.verbose >= 1:
            print(*args, **kwargs)

    @property
    def model(self):
        return self._model
