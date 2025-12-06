"""Implementation inspired by https://github.com/jlparkI/mix_T/blob/main/src/studenttmixture/em_student_mixture.py"""

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln, logsumexp
from sklearn.base import BaseEstimator, ClusterMixin, DensityMixin
from sklearn.cluster import KMeans
from tqdm import trange

EPS = np.finfo(float).eps


def _maha_dist(scale_cholesky_, loc_, X):
    diff = X - loc_[None, :]
    diff = solve_triangular(scale_cholesky_, diff.T, lower=True)
    return (diff**2).sum(axis=0)


def sq_maha_distance(X, loc_, scale_cholesky_):
    sc = jnp.transpose(scale_cholesky_, (-1, 0, 1))
    v_f = jax.vmap(partial(_maha_dist, X=X), in_axes=(0, 0), out_axes=0)
    return v_f(sc, loc_).T


def _scale_update(ru, loc_, resp_sum, reg_covar, X):
    diff = X - loc_[None, :]
    scale = jnp.dot(ru * diff.T, diff) / (resp_sum + 10 * EPS)
    scale = scale + reg_covar * jnp.eye(scale.shape[0])
    return scale


def scale_update_calcs(X, ru, loc_, resp_sum, reg_covar):
    """Updates the scale (aka covariance) matrices as part of the M-
    step for EM and as the parameter update for variational methods."""
    ru = ru.T
    v_f = jax.vmap(
        partial(_scale_update, reg_covar=reg_covar, X=X),
        in_axes=(0, 0, 0),
        out_axes=(0),
    )
    scale = v_f(ru, loc_, resp_sum)
    scale_cholesky = jax.vmap(jnp.linalg.cholesky, 0, 0)(scale)
    return scale.transpose(1, 2, 0), scale_cholesky.transpose(1, 2, 0)


def logdet(a):
    return jnp.sum(jnp.log(jnp.diag(a)))


def get_loglikelihood(X, sq_maha_dist, df_, scale_cholesky_, mix_weights_):
    sq_maha_dist = 1 + sq_maha_dist / df_[jnp.newaxis, :]
    sq_maha_dist = -0.5 * (df_[jnp.newaxis, :] + X.shape[1]) * jnp.log(sq_maha_dist)
    const_term = gammaln(0.5 * (df_ + X.shape[1])) - gammaln(0.5 * df_)
    const_term = const_term - 0.5 * X.shape[1] * (jnp.log(df_) + jnp.log(jnp.pi))
    scale_logdet = jax.vmap(logdet, 0, 0)(jnp.transpose(scale_cholesky_, (-1, 0, 1)))
    return -scale_logdet[jnp.newaxis, :] + const_term[jnp.newaxis, :] + sq_maha_dist


def _e_step(X, df_, loc_, scale_cholesky_, mix_weights_, sq_maha_dist):
    sq_maha_dist = sq_maha_distance(X, loc_, scale_cholesky_)
    loglik = get_loglikelihood(X, sq_maha_dist, df_, scale_cholesky_, mix_weights_)
    weighted_log_prob = (
        loglik
        + jnp.log(jnp.clip(mix_weights_, a_min=1e-12, a_max=None))[jnp.newaxis, :]
    )
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    resp = jnp.exp(weighted_log_prob - log_prob_norm[:, jnp.newaxis])
    E_gamma = (df_[jnp.newaxis, :] + X.shape[1]) / (df_[jnp.newaxis, :] + sq_maha_dist)
    lower_bound = jnp.mean(log_prob_norm)
    return resp, E_gamma, lower_bound


def _m_step(X, resp, E_gamma, scale_, scale_cholesky_, df_, reg_covar):
    mix_weights_ = jnp.mean(resp, axis=0)
    ru = resp * E_gamma
    loc_ = jnp.dot(ru.T, X)
    resp_sum = jnp.sum(ru, axis=0) + 10 * EPS
    loc_ = loc_ / resp_sum[:, jnp.newaxis]
    scale_, scale_cholesky_ = scale_update_calcs(X, ru, loc_, resp_sum, reg_covar)
    return mix_weights_, loc_, scale_, scale_cholesky_


class StudentsTMixture(BaseEstimator, ClusterMixin, DensityMixin):
    """Student's T Mixture Model.
    This class allows you to estimate a mixture of multivariate t-distributions over your data.

    Parameters
    ----------
    n_components: int
        The number of mixture components.
    tol: float, default=1e-5
        The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
    reg_covar: float, default=1e-6
        Non-negative regularization added to the diagonal of covariance. Allows to assure that the covariance matrices are all positive.
    max_iter: int, default=1000
        The number of EM iterations to perform.
    df: float, default=4.0
        Degrees of freedom for the t-Distributions.
    random_state: int, default=None
        Random state for reproducibility.

    Attributes
    ----------
    weights_: array-like of shape (n_components,)
        The weights of each mixture components.
    means_: array-like of shape (n_components, n_features)
        The mean of each mixture component.
    n_iter_: int
        Number of step used by the best fit of EM to reach the convergence.
    converged_: bool
        True when convergence of the best fit of EM was reached, False otherwise.
    """

    def __init__(
        self,
        n_components: int,
        tol: float = 1e-5,
        reg_covar: float = 1e-06,
        max_iter: int = 1000,
        df=4.0,
        random_state=None,
    ):
        super().__init__()
        self.start_df_ = float(df)
        self.n_components = n_components
        self.tol = tol
        self.df = df
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state

    def _init_params(self, X):
        loc_ = (
            KMeans(self.n_components, random_state=self.random_state)
            .fit(X)
            .cluster_centers_
        )
        mix_weights_ = np.ones(self.n_components) / self.n_components
        default_scale_matrix = np.cov(X, rowvar=False)
        default_scale_matrix.flat[:: X.shape[1] + 1] += self.reg_covar
        if len(default_scale_matrix.shape) < 2:
            default_scale_matrix = default_scale_matrix.reshape(-1, 1)
        scale_ = np.stack(
            [default_scale_matrix for i in range(self.n_components)], axis=-1
        )
        scale_cholesky_ = [
            np.linalg.cholesky(scale_[:, :, i]) for i in range(self.n_components)
        ]
        scale_cholesky_ = np.stack(scale_cholesky_, axis=-1)
        return loc_, scale_, mix_weights_, scale_cholesky_

    @staticmethod
    def e_step(X, df_, loc_, scale_cholesky_, mix_weights_, sq_maha_dist):
        return _e_step(X, df_, loc_, scale_cholesky_, mix_weights_, sq_maha_dist)

    @staticmethod
    def m_step(X, resp, E_gamma, scale_, scale_cholesky_, df_, reg_covar):
        return _m_step(X, resp, E_gamma, scale_, scale_cholesky_, df_, reg_covar)

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to a single data point.

        y: Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self: StudentsTMixture
            The fitted mixture.
        """
        self.df_ = np.full((self.n_components), self.df, dtype=np.float64)
        loc_, scale_, mix_weights_, scale_cholesky_ = self._init_params(X)
        lower_bound = -np.inf
        sq_maha_dist = np.empty((X.shape[0], self.n_components))
        self.converged_ = True

        @jax.jit
        def step(loc_, mix_weights_, scale_cholesky_, scale_, sq_maha_dist):
            resp, E_gamma, current_bound = self.e_step(
                X, self.df_, loc_, scale_cholesky_, mix_weights_, sq_maha_dist
            )
            mix_weights_, loc_, scale_, scale_cholesky_ = self.m_step(
                X,
                resp,
                E_gamma,
                scale_,
                scale_cholesky_,
                self.df_,
                self.reg_covar,
            )
            return (
                loc_,
                mix_weights_,
                scale_cholesky_,
                scale_,
                sq_maha_dist,
                current_bound,
            )

        for iter in trange(self.max_iter, desc="Running EM"):
            loc_, mix_weights_, scale_cholesky_, scale_, sq_maha_dist, current_bound = (
                step(
                    loc_,
                    mix_weights_,
                    scale_cholesky_,
                    scale_,
                    sq_maha_dist,
                )
            )
            change = current_bound - lower_bound
            if abs(change) < self.tol:
                break
            lower_bound = current_bound
        else:
            self.converged_ = False
            warnings.warn(
                "tDistributionMixture did not converge, consider increasing the number of iterations."
            )
        self.n_iter_ = iter
        self.weights_ = mix_weights_
        self.means_ = loc_
        self.scale_cholesky_ = scale_cholesky_
        self.scale_ = scale_
        return self

    def predict_proba(self, X):
        """Evaluate the components' density for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each Student's T component for each sample in X.
        """
        sq_maha_dist = sq_maha_distance(X, self.means_, self.scale_cholesky_)
        loglik = get_loglikelihood(
            X, sq_maha_dist, self.df_, self.scale_cholesky_, self.weights_
        )
        weighted_loglik = loglik + np.log(self.weights_)[np.newaxis, :]
        with np.errstate(under="ignore"):
            loglik = weighted_loglik - logsumexp(weighted_loglik, axis=1)[:, np.newaxis]
        return np.exp(loglik)

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def fit_predict(self, X):
        """Estimate model parameters using X and predict the labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        return self.fit(X).predict(X)

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X: array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        aic: float
            The lower the better.
        """
        self.check_model()
        x = self.check_inputs(X)
        n_params = self.get_num_parameters()
        score = self.score(x, perform_model_checks=False)
        return 2 * n_params - 2 * score * X.shape[0]

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X: array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        bic: float
            The lower the better.
        """
        self.check_model()
        x = self.check_inputs(X)
        score = self.score(x, perform_model_checks=False)
        n_params = self.get_num_parameters()
        return n_params * np.log(x.shape[0]) - 2 * score * x.shape[0]

    def score_samples(self, X):
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        sq_maha_dist = sq_maha_distance(X, self.means_, self.scale_cholesky_)
        loglik = get_loglikelihood(
            X, sq_maha_dist, self.df_, self.scale_cholesky_, self.weights_
        )
        weighted_loglik = loglik + np.log(self.weights_)[np.newaxis, :]
        return logsumexp(weighted_loglik, axis=1)

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of `X` under the Gaussian mixture model.
        """
        return np.mean(self.score_samples(X))


class CauchyMixture(StudentsTMixture):
    """Cauchy Mixture Model.
    This class allows you to estimate a multivariate mixture of Cauchys over your data.
    Equivalent to StudentsTMixture, except the degrees of freedom is fixed to 1.

    Parameters
    ----------
    n_components: int
        The number of mixture components.
    tol: float, default=1e-5
        The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
    reg_covar: float, default=1e-6
        Non-negative regularization added to the diagonal of covariance. Allows to assure that the covariance matrices are all positive.
    max_iter: int, default=1000
        The number of EM iterations to perform.

    Attributes
    ----------
    weights_: array-like of shape (n_components,)
        The weights of each mixture components.
    means_: array-like of shape (n_components, n_features)
        The mean of each mixture component.
    n_iter_: int
        Number of step used by the best fit of EM to reach the convergence.
    converged_: bool
        True when convergence of the best fit of EM was reached, False otherwise.
    """

    def __init__(
        self,
        n_components: int,
        tol: float = 1e-5,
        reg_covar: float = 1e-06,
        max_iter: int = 1000,
        random_state=None,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            random_state=random_state,
            df=1.0,
        )
