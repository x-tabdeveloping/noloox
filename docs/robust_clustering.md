# Outlier-Robust Clustering

While Gaussian Mixtures and KMeans can be powerful clustering models, their estimates can be highly affected by the presence of outliers.
In this tutorial we will look at how to use a [mixture of multivariate Student's T distributions](StudentsTMixture.md) to cluster observations, which is much more robust to outliers due to heavier tails in the distributions.

### Data simulation

We will first generate a dataset, in which there are two clear clusters, and one of the clusters is going to contain a number of outliers, that might bias our estimate if we are not using a robust model.
We will use scikit-learn's `make_blobs` convenience function for this:

```python
from sklearn.datasets import make_blobs
import numpy as np

X, y_true = make_blobs(centers=[[2, 2], [-2, -2]], n_samples=1000, cluster_std=0.5)
# Adding 50 outliers:
X_outliers, y_outliers = make_blobs(centers=[[5, 5]], n_samples=50, cluster_std=0.1)
X = np.concatenate([X, X_outliers])
y_true = np.concatenate([y_true, y_outliers])
```

Let us plot the data to get a feel for how it looks:

```python
import plotly.express as px

fig = px.scatter(x=X[:, 0], y=X[:, 1], color=list(map(str, y_true)))
fig.show()
```

<figure>
  <iframe src="../images/blobs_w_outliers.html", title="Blobs with outliers", style="height:600px;width:800px;padding:0px;border:none;"></iframe>
  <figcaption> Two clear clusters with 50 outliers in one of them. </figcaption>
</figure>

We see that cluster 0 has some outliers, that might bias our estimate.

### Model fitting

Let's fit a Gaussian Mixture model and a Student's t Mixture model to our data and save the predictions:

```python
from sklearn.mixture import GaussianMixture
from noloox.mixture import StudentsTMixture

gmm = GaussianMixture(2)
y_gaussian = gmm.fit_predict(X)

stmm = StudentsTMixture(2, df=1.0)
y_t = stmm.fit_predict(X)
```

### Model criticism

To see how well the models' predictions align with or true labels, we can look at the Adjusted Rand Index between the true and predicted clusterings.

```python
from sklearn.metrics import adjusted_rand_score

print("ARI(Gaussian): ", adjusted_rand_score(y_true, y_gaussian))
print("ARI(Student's T): ", adjusted_rand_score(y_true, y_t))
```

```
ARI(Gaussian):  1.0
ARI(Student's T):  1.0
```

We see that both models' labels perfectly match the true labels, meaning, they identified the correct clusters.

Now let's examine how close each models' estimate lies to the true cluster centres:

```python
import plotly.graph_objects as go

fig = go.Figure()
fig = fig.add_scatter(
    x=X[:, 0],
    y=X[:, 1],
    mode="markers",
    marker=dict(size=5, color="black"),
    opacity=0.5,
    showlegend=False,
)
fig = fig.add_scatter(
    x=gmm.means_[:, 0],
    y=gmm.means_[:, 1],
    name="GMM cluster centres",
    mode="markers",
    marker=dict(
        color="white", size=15, symbol="cross", line=dict(width=2, color="black")
    ),
    zorder=10,
)
fig = fig.add_scatter(
    x=stmm.means_[:, 0],
    y=stmm.means_[:, 1],
    name="StudentsTMixture cluster centres",
    mode="markers",
    marker=dict(color="white", size=15, symbol="x", line=dict(width=2, color="black")),
    zorder=10,
)
fig = fig.add_scatter(
    x=[2, -2],
    y=[2, -2],
    name="True cluster centers",
    mode="markers",
    marker=dict(
        color="white", size=15, symbol="star", line=dict(width=2, color="black")
    ),
    zorder=10,
)
fig.show()
```

<figure>
  <iframe src="../images/robust_cluster_means.html", title="Robust cluster means", style="height:950px;width:1200px;padding:0px;border:none;"></iframe>
  <figcaption> Cluster centroids identified by GMM and StudentsTMixture compared to true cluster means. </figcaption>
</figure>

We can see that StudentsTMixture identified the cluster centres much more faithfully. The Gaussian Mixture model's estimate of the cluster means has been substantially affected by the presence of outliers

