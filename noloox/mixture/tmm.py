"""Implementation inspired by https://github.com/jlparkI/mix_T/blob/main/src/studenttmixture/em_student_mixture.py"""

import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp
from sklearn.base import BaseEstimator, ClusterMixin, DensityMixin
from sklearn.cluster import KMeans


def sq_maha_distance(X, loc_, scale_cholesky_):
    sq_maha_dist = np.empty((X.shape[0], loc_.shape[0]))
    for i in range(loc_.shape[0]):
        diff = X - loc_[None, i, :]
        diff = solve_triangular(scale_cholesky_[:, :, i], diff.T, lower=True)
        sq_maha_dist[:, i] = (diff**2).sum(axis=0)
    return sq_maha_dist


def scale_update_calcs(X, ru, loc_, resp_sum, reg_covar):
    """Updates the scale (aka covariance) matrices as part of the M-
    step for EM and as the parameter update for variational methods."""
    scale_ = np.empty((loc_.shape[1], loc_.shape[1], loc_.shape[0]))
    scale_cholesky_ = np.empty((loc_.shape[1], loc_.shape[1], loc_.shape[0]))
    for i in range(loc_.shape[0]):
        diff = X - loc_[i : i + 1, :]
        scale_[:, :, i] = np.dot(ru[:, i] * diff.T, diff) / (
            resp_sum[i] + 10 * np.finfo(scale_.dtype).eps
        )
        scale_[:, :, i].flat[:: scale_.shape[0] + 1] += reg_covar
        scale_cholesky_[:, :, i] = np.linalg.cholesky(scale_[:, :, i])
    return scale_, scale_cholesky_


def logdet(a):
    return np.sum(np.log(np.diag(a)))


def get_loglikelihood(X, sq_maha_dist, df_, scale_cholesky_, mix_weights_):
    sq_maha_dist = 1 + sq_maha_dist / df_[np.newaxis, :]
    sq_maha_dist = -0.5 * (df_[np.newaxis, :] + X.shape[1]) * np.log(sq_maha_dist)
    const_term = gammaln(0.5 * (df_ + X.shape[1])) - gammaln(0.5 * df_)
    const_term = const_term - 0.5 * X.shape[1] * (np.log(df_) + np.log(np.pi))
    scale_logdet = [
        logdet(scale) for scale in np.transpose(scale_cholesky_, (-1, 0, 1))
    ]
    scale_logdet = np.asarray(scale_logdet)
    return -scale_logdet[np.newaxis, :] + const_term[np.newaxis, :] + sq_maha_dist


def e_step(X, df_, loc_, scale_cholesky_, mix_weights_, sq_maha_dist):
    sq_maha_dist = sq_maha_distance(X, loc_, scale_cholesky_)
    loglik = get_loglikelihood(X, sq_maha_dist, df_, scale_cholesky_, mix_weights_)
    weighted_log_prob = (
        loglik + np.log(np.clip(mix_weights_, a_min=1e-12, a_max=None))[np.newaxis, :]
    )
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under="ignore"):
        resp = np.exp(weighted_log_prob - log_prob_norm[:, np.newaxis])
    E_gamma = (df_[np.newaxis, :] + X.shape[1]) / (df_[np.newaxis, :] + sq_maha_dist)
    lower_bound = np.mean(log_prob_norm)
    return resp, E_gamma, lower_bound


def m_step(X, resp, E_gamma, scale_, scale_cholesky_, df_, reg_covar):
    mix_weights_ = np.mean(resp, axis=0)
    ru = resp * E_gamma
    loc_ = np.dot(ru.T, X)
    resp_sum = np.sum(ru, axis=0) + 10 * np.finfo(resp.dtype).eps
    loc_ = loc_ / resp_sum[:, np.newaxis]
    scale_, scale_cholesky_ = scale_update_calcs(X, ru, loc_, resp_sum, reg_covar)
    return mix_weights_, loc_, scale_, scale_cholesky_, df_


class StudentsTMixture(BaseEstimator, ClusterMixin, DensityMixin):
    """Student's T Mixture Model.
    This class allows you to estimate a TMM over your data.

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
        df=4.0,
        random_state=None,
        verbose=False,
    ):
        super().__init__()
        self.start_df_ = float(df)
        self.n_components = n_components
        self.tol = tol
        self.df = df
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

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

    def fit(self, X, y=None):
        """"""
        self.df_ = np.full((self.n_components), self.df, dtype=np.float64)
        loc_, scale_, mix_weights_, scale_cholesky_ = self._init_params(X)
        lower_bound = -np.inf
        sq_maha_dist = np.empty((X.shape[0], self.n_components))
        self.converged_ = True
        for iter in range(self.max_iter):
            resp, E_gamma, current_bound = e_step(
                X, self.df_, loc_, scale_cholesky_, mix_weights_, sq_maha_dist
            )

            mix_weights_, loc_, scale_, scale_cholesky_, df_ = m_step(
                X,
                resp,
                E_gamma,
                scale_,
                scale_cholesky_,
                self.df_,
                reg_covar=self.reg_covar,
            )
            change = current_bound - lower_bound
            if abs(change) < self.tol:
                break
            lower_bound = current_bound
            if self.verbose:
                print(f"Change in lower bound: {change}")
                print(f"Actual lower bound: {current_bound}")
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
        sq_maha_dist = sq_maha_distance(X, self.means_, self.scale_cholesky_)
        loglik = get_loglikelihood(
            X, sq_maha_dist, self.df_, self.scale_cholesky_, self.weights_
        )
        weighted_loglik = loglik + np.log(self.weights_)[np.newaxis, :]
        with np.errstate(under="ignore"):
            loglik = weighted_loglik - logsumexp(weighted_loglik, axis=1)[:, np.newaxis]
        return np.exp(loglik)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit_predict(self, X):
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
        sq_maha_dist = sq_maha_distance(X, self.means_, self.scale_cholesky_)
        loglik = get_loglikelihood(
            X, sq_maha_dist, self.df_, self.scale_cholesky_, self.weights_
        )
        weighted_loglik = loglik + np.log(self.weights_)[np.newaxis, :]
        return logsumexp(weighted_loglik, axis=1)

    def score(self, X):
        return np.mean(self.score_samples(X))

    def sample(self, num_samples: int = 1):
        rng = np.random.default_rng(self.random_state)
        labels = rng.multinomial(1, self.weights_, size=num_samples)
        X_gen = []
        for label in labels:
            pass
