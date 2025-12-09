
# 

<img src="images/icon.png" width="75px"></img> is a Python library containing reference implementations of a bunch of very useful unsupervised learning algorithms that you probably won't find elsewhere.

#### What <img src="images/icon.png" width="75px"></img> is:

- A collection of unsupervised machine learning algorithms
- A scikit-learn compatible library
- An educational resource containing worked examples and reference implementation

#### What <img src="images/icon.png" width="75px"></img> isn't:

- The most feature-complete or efficient implementation of these algorithms
- A replacement for scikit-learn
- An all-in-one machine learning framework

## Basic usage

Install noloox from PyPI:

```bash
pip install noloox
```

Then you can load models from the library and use them the same way you would use scikit-learn.

```python
from noloox.mixture import StudentsTMixture

model = StudentsTMixture(n_components=10)
cluster_labels = model.fit_predict(X)
```

## Models

| Model | What do I use it for? | JAX or NumPy? | What algorithm? | Tutorial |
| - | - | - | - | - |
| [Peax](Peax.md) | Cluster 2D data where the number of clusters is unknown. | NumPy | Expectation-Maximization | [Finding the number of clusters in the data](n_clusters.md) |
| [SNMF](SNMF.md) | Factor data, where you expect the factors to be non-negative, but the data is unbounded | JAX | Iterative updates | [Topic discovery by factoring transformer embeddings](senstopic.md) |
| [WNMF](WNMF.md) | NMF, but you don't want to weight all observations equally. | NumPy | Iterative updates | - |
| [StudentsTMixture](StudentsTMixture.md)/[CauchyMixture](CauchyMixture.md) | Cluster continuous data in a way that is robust to outliers. | JAX | Expectation-Maximization | [Outlier-Robust Clustering](robust_clustering.md) |
| [DirichletMultinomialMixture](DirichletMultinomialMixture.md) | Cluster count data/Short-text topic modelling | JAX | Collapsed Gibbs Sampling | [Topic modelling for short texts](short_topic_modelling.md) and [Clustering Count Data](count_clustering.md) |

## Our philosophy and goals

 - Keep implementations simple and minimal, Minimal dependencies
 - Everything should either be implemented in NumPy or JAX. Preferably as many in JAX as possible.
 - Library structure should match sklearn standards, and all algorithms should be drop-in replacements for scikit-learn equivalents.
 - Under these restrictions, algorithms should be as fast as humanly possible

## The <img src="images/icon.png" width="90px"></img> wishlist:

There are a number of algorithms that would be nice to implement in the library.
Contributions are very welcome.

 - ProdLDA, and amortized ProdLDA (CTMs) (without Flax)
 - Parametric-TSNE, possibly also Multi-scale Parametric-TSNE
 - DiRE
 - Infinite NMF
 - Latent Dirichlet Allocation with Gibbs Sampling
 - Gaussian LDA

