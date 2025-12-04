from typing import Optional

import numpy as np
from scipy.ndimage import (binary_erosion, generate_binary_structure,
                           maximum_filter)
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (_compute_precision_cholesky,
                                               _estimate_gaussian_parameters)


def detect_peaks(image):
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 25)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # we create the mask of the background
    background = image == 0
    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks


class FixedMeanGaussianMixture(GaussianMixture):
    def _m_step(self, X, log_resp):
        # Skipping mean update
        self.weights_, _, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )


class Peax(ClusterMixin, BaseEstimator):
    """Peax clustering model.
    The model estimates the number of clusters from density peaks,
    then uses Gaussian Mixtures with fixed means to estimate cluster
    probabilities.

    Parameters
    ----------
    random_state: int, default None
        Random seed to use for fitting gaussian mixture to peaks.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in `X`.
    gmm_ : FixedMeanGaussianMixture
        The fitted Gaussian mixture model with fixed means.
    means_ : ndarray of shape (n_components, 2)
        Coordinates of detected density peaks used as cluster means.
    weights_ : ndarray of shape (n_components,)
        Final mixture component weights after refitting.
    classes_ : ndarray of shape (n_components,)
        Sorted array of unique cluster labels.
    density : gaussian_kde
        Kernel density estimator fitted to the input data.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def fit(self, X, y=None):
        if X.shape[1] > 2:
            raise ValueError(
                f"X has {X.shape[1]} > 2 features. Peax only accepts 2D data."
            )
        self.X_range = np.min(X), np.max(X)
        self.density = gaussian_kde(X.T, "scott")
        coord = np.linspace(*self.X_range, num=100)
        z = []
        for yval in coord:
            points = np.stack([coord, np.full(coord.shape, yval)]).T
            prob = np.exp(self.density.logpdf(points.T))
            z.append(prob)
        z = np.stack(z)
        peaks = detect_peaks(z.T)
        peak_ind = np.nonzero(peaks)
        peak_pos = np.stack([coord[peak_ind[0]], coord[peak_ind[1]]]).T
        weights = self.density.pdf(peak_pos.T)
        weights = weights / weights.sum()
        self.gmm_ = FixedMeanGaussianMixture(
            peak_pos.shape[0],
            means_init=peak_pos,
            weights_init=weights,
            random_state=self.random_state,
        )
        self.labels_ = self.gmm_.fit_predict(X)
        # Checking whether there are close to zero components
        is_zero = np.isclose(self.gmm_.weights_, 0)
        n_zero = np.sum(is_zero)
        if n_zero > 0:
            print(f"{n_zero} components have zero weight, removing them and refitting.")
        peak_pos = peak_pos[~is_zero]
        weights = self.gmm_.weights_[~is_zero]
        weights = weights / weights.sum()
        self.gmm_ = FixedMeanGaussianMixture(
            peak_pos.shape[0],
            means_init=peak_pos,
            weights_init=weights,
            random_state=self.random_state,
        )
        self.labels_ = self.gmm_.fit_predict(X)
        self.classes_ = np.sort(np.unique(self.labels_))
        self.means_ = self.gmm_.means_
        self.weights_ = self.gmm_.weights_
        return self.labels_

    @property
    def n_components(self) -> int:
        return self.gmm_.n_components

    def predict_proba(self, X):
        return self.gmm_.predict_proba(X)

    def score_samples(self, X):
        return self.density.logpdf(X.T)

    def score(self, X):
        return np.mean(self.score_samples(X))
