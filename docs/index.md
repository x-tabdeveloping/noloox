
# noloox

noloox is a Python library containing reference implementations of a bunch of very useful unsupervised learning algorithms that you probably won't find elsewhere.

### Our philosophy and goals

 - Keep implementations simple and minimal
 - Minimal dependencies
 - Everything should either be implemented in NumPy or JAX
 - Library structure should match sklearn standards, and all algorithms should be drop-in replacements for scikit-learn equivalents.
 - Under these restrictions, algorithms should be as fast as humanly possible

### Basic usage

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
