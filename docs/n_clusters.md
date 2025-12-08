# Finding the number of clusters in the data

While most of the time, it is hard to make an educated guess about how many clusters there might be in a dataset ahead of time,
not all clustering models can, detect the number of clusters that is in the underlying data.
In this tutorial we will look at the [Peax](Peax.md) clustering model,
that determines the number of clusters in a dataset based on how many peaks there are in the data's Kernel Density Estimate.

!!! warning
    Peax, at the moment is only able to cluster data in two-dimensions.
    If you intend to use it with higher-dimensional data, we recommend that you first reduce the number of dimensions using [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

### Data simulation

We will simulate datasets with varying numbers of clusters to see how well the model can detect the number of clusters in the dataset.
We will use scikit-learn's `make_blobs` utility for this.

```python
import numpy as np
from sklearn.datasets import make_blobs

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
```

Let's plot one of them to see how this roughly looks:

```python
import plotly.express as px

px.scatter(x=Xs[4][:, 0], y=Xs[4][:, 1], color=ys[4])
```

<figure>
  <iframe src="../images/5_blobs.html", title="5 Blobs", style="height:600px;width:800px;padding:0px;border:none;"></iframe>
  <figcaption> 5 randomly generated blobs with random cluster centres.. </figcaption>
</figure>

### Recovering the clusters

We will run a Peax model for each of the generated datasets, and then print the Adjusted Rand Index, to see how well the original clustering was recovered. We will also save results into a data frame so we can plot later:

```python
from noloox.cluster import Peax
from sklearn.metrics import adjusted_rand_score
import pandas as pd

records = []
for X, y, n in zip(Xs, ys, n_clusters):
    model = Peax(random_state=42)
    y_pred = model.fit_predict(X)
    n_pred = model.n_components
    ari = adjusted_rand_score(y, y_pred)
    print(f"N={n}; N_pred={n_pred}; ARI={ari}")
    for (x0, x1), pred_label in zip(X, y_pred):
        records.append(dict(x0=x0, x1=x1, label=str(pred_label), n=n))
res_df = pd.DataFrame.from_records(records)
```

```
N=1; N_pred=1; ARI=1.0
N=2; N_pred=2; ARI=0.28885891668204566
N=3; N_pred=3; ARI=1.0
N=4; N_pred=4; ARI=1.0
N=5; N_pred=5; ARI=1.0
N=6; N_pred=5; ARI=0.8231137071695086
N=7; N_pred=5; ARI=0.7299110696658858
N=8; N_pred=6; ARI=0.6741328694139394
N=9; N_pred=7; ARI=0.7897171474615344
```

We see that we recover both the number of clusters and cluster membership quite well.
It seems that the most problematic one is when we have two clusters.

Let's investigate the results visually:

```python
px.scatter(res_df, x="x0", y="x1", color="label", facet_col="n", facet_col_wrap=3)
```

<figure>
  <iframe src="../images/peax.html", title="Clusters recovered by Peax", style="height:800px;width:1000px;padding:0px;border:none;"></iframe>
  <figcaption> Clusters recovered by Peax in all datasets. </figcaption>
</figure>

We see that cluster recovery only failed when two clusters were particularly merged together or very close to each other.
Now that we see the results, it also makes sense why the model failed on N=2, where the two clusters are merged together.
Impressively enough the model still detected that there were two clusters here.
