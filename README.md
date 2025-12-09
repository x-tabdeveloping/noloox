
<img src="docs/images/icon.png" width="75px"></img> is a Python library containing reference implementations of a bunch of very useful unsupervised learning algorithms that you probably won't find elsewhere.

#### What <img src="docs/images/icon.png" width="75px"></img> is:

- A collection of unsupervised machine learning algorithms
- A scikit-learn compatible library
- An educational resource containing worked examples and reference implementation

#### What <img src="docs/images/icon.png" width="75px"></img> isn't:

- The most feature-complete or efficient implementation of these algorithms
- A replacement for scikit-learn
- An all-in-one machine learning framework
- A library for complete Bayesian inference. Use a PPL like NumPyro, PyMC or Stan.

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

| Model | What do I use it for? | JAX or NumPy? | What algorithm? |
| - | - | - | - |
| [Peax](https://x-tabdeveloping.github.io/noloox/Peax/) | Cluster 2D data where the number of clusters is unknown. | NumPy | Expectation-Maximization | 
| [SNMF](https://x-tabdeveloping.github.io/noloox/SNMF/) | Factor data, where you expect the factors to be non-negative, but the data is unbounded | JAX | Iterative updates |
| [WNMF](https://x-tabdeveloping.github.io/noloox/WNMF/) | NMF, but you don't want to weight all observations equally. | NumPy | Iterative updates |
| [StudentsTMixture](https://x-tabdeveloping.github.io/noloox/StudentsTMixture/) and [CauchyMixture](https://x-tabdeveloping.github.io/noloox/CauchyMixture/) | Cluster continuous data in a way that is robust to outliers. | JAX | Expectation-Maximization |
| [DirichletMultinomialMixture](https://x-tabdeveloping.github.io/noloox/DirichletMultinomialMixture/) | Cluster count data/Short-text topic modelling | JAX | Collapsed Gibbs Sampling |

## Tutorials

Here are some things, that you can do in easier in <img src="docs/images/icon.png" width="75px"></img> than in scikit-learn:

- [Cluster count data](https://x-tabdeveloping.github.io/noloox/count_clustering/)
- [Topic modelling for short texts](https://x-tabdeveloping.github.io/noloox/short_topic_modelling/)
- [Clustering robust to outliers](https://x-tabdeveloping.github.io/noloox/robust_clustering/)
- [Find the number of clusters in your dataset](https://x-tabdeveloping.github.io/noloox/n_clusters/)
- [Find nonnegative factors in sentence embeddings](https://x-tabdeveloping.github.io/noloox/senstopic/)

## Our philosophy and goals

 - Keep implementations simple and minimal, Minimal dependencies
 - Everything should either be implemented in NumPy or JAX. Preferably as many in JAX as possible.
 - Library structure should match sklearn standards, and all algorithms should be drop-in replacements for scikit-learn equivalents.
 - Under these restrictions, algorithms should be as fast as humanly possible

## The <img src="docs/images/icon.png" width="90px"></img> wishlist:

There are a number of algorithms that would be nice to implement in the library.
Contributions are very welcome.

 - ProdLDA, and amortized ProdLDA (CTMs) (without Flax)
 - Parametric-TSNE, possibly also Multi-scale Parametric-TSNE
 - DiRE
 - Infinite NMF
 - Latent Dirichlet Allocation with Gibbs Sampling
 - Gaussian LDA

