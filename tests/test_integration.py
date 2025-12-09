import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture

from noloox.cluster import Peax
from noloox.decomposition import SNMF, WNMF
from noloox.mixture import DirichletMultinomialMixture, StudentsTMixture


def test_peax():
    rng = np.random.default_rng(42)
    n_clusters = list(range(1, 10))
    Xs = []
    ys = []
    for n in n_clusters:
        centers = rng.normal(0, 10, size=(n, 2))
        X, y_true = make_blobs(
            centers=centers, n_samples=1000, cluster_std=0.5, random_state=0
        )
        Xs.append(X)
        ys.append(y_true)
    aris = []
    for X, y, n in zip(Xs, ys, n_clusters):
        model = Peax(random_state=42)
        y_pred = model.fit_predict(X)
        ari = adjusted_rand_score(y, y_pred)
        aris.append(ari)
    assert np.mean(ari) > 0.5, "Peax does not function properly, produces poor results."


def test_students_t_mixture():
    true_means = np.array([[2, 2], [-2, -2]])
    X, y_true = make_blobs(
        centers=true_means, n_samples=1000, cluster_std=0.5, random_state=42
    )
    X_outliers, y_outliers = make_blobs(
        centers=[[5, 5]], n_samples=50, cluster_std=0.1, random_state=42
    )
    X = np.concatenate([X, X_outliers])
    y_true = np.concatenate([y_true, y_outliers])
    stmm = StudentsTMixture(2, df=1.0, random_state=42)
    y_t = stmm.fit_predict(X)
    assert (
        adjusted_rand_score(y_true, y_t) > 0.5
    ), "StudentsTMixture does not function as it should."
    gmm = GaussianMixture(2, random_state=42)
    gmm.fit(X)
    gmm_err = euclidean_distances(gmm.means_, true_means).min(axis=1).mean()
    stmm_err = euclidean_distances(stmm.means_, true_means).min(axis=1).mean()
    assert gmm_err > stmm_err, "GMM is less affected by outliers than StudentsTMixture"


def test_dmm():
    generator = np.random.default_rng(42)
    columns = [
        "engineers",
        "academics",
        "doctors",
        "lawyers",
        "artists",
        "musicians",
        "skilled worker",
        "other",
    ]
    true_prob = [
        [0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05],  # Academic orientation
        [0.4, 0, 0.05, 0.05, 0.0, 0.05, 0.3, 0.15],  # Technical orientation
        [0, 0.1, 0.05, 0.05, 0.3, 0.4, 0, 0.1],  # Art orientation
    ]
    weights = [0.2, 0.7, 0.1]
    n_schools = 200
    n_students_per_school = 30
    X = []
    y_true = []
    for i in range(n_schools):
        school_type = generator.choice(np.arange(3), p=weights)
        y_true.append(school_type)
        school_graduates = np.zeros(len(columns), dtype="int")
        for j in range(n_students_per_school):
            profession = generator.choice(
                np.arange(len(columns)), p=true_prob[school_type]
            )
            school_graduates[profession] += 1
        X.append(school_graduates)
    X = np.stack(X)
    y_true = np.array(y_true)
    model = DirichletMultinomialMixture(n_components=3, random_state=42)
    y_pred = model.fit_predict(X)
    assert (
        adjusted_rand_score(y_true, y_pred) > 0.5
    ), "DirichletMultinomialMixture does not function as it should"


def test_snmf():
    rng = np.random.default_rng(42)
    X = np.concatenate(
        [rng.normal(0, 1, size=(200, 10)), rng.normal(-5, 1, size=(100, 10))]
    )
    model = SNMF(2, sparsity=1.0, random_state=42)
    X_transformed = model.fit_transform(X)
    assert np.all(X_transformed) >= 0, "SNMF: Not all entries are non-negative"


def test_wnmf():
    rng = np.random.default_rng(42)
    X = np.concatenate(
        [rng.normal(0, 1, size=(200, 10)), rng.normal(-2, 1, size=(100, 10))]
    )
    X = np.square(X)
    weights = np.abs(rng.normal(1, 0.5, size=(300)))
    model = WNMF(2).fit(X, weights)
