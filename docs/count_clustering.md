# Clustering Count Data

While algorithms like KMeans and Gaussian Mixtures might be perfectly suitable for clustering continuous and normally distributed data,
they are not an appropriate choice for clustering multivariate count data.

In this example we will generate synthetic count data with underlying true clusters, and then look at how to recover these clusters completely unsupervised using [`DirichletMultinomialMixture`](DirichletMultinomialMixture.md).

### Simulating data

In our example we will generate data about where students from different high-schools in a region ended up after having graduated from the school.
Let's say that we have 200 schools, and each of the schools has thirty students.
Every student that graduates has some probability of becoming and engineer, and academic, a doctor, a lawyer, an artist, a musician, a skilled worker, or something else.

We will assume that there are three underlying clusters in this data, and will use this assumption to generate stochastic data from these three types of schools.

1. 20% of schools have an academic orientation, where most graduates become engineers, academics, doctors or lawyers.
2. 10% of schools have an art orientation, where most graduates become artists and musicians with some probability of becoming an academic or something else.
3. 70% of schools have a technical orientation, where most graduates go on to be engineers or skilled workers.

We will simulate the number of graduates going into any one of the listed professions from each school using NumPy.

```python
import numpy as np

generator = np.random.default_rng(42)

columns = ["engineers", "academics", "doctors", "lawyers", "artists", "musicians", "skilled worker", "other"]
true_prob = [
    [0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05], # Academic orientation
    [0.4, 0, 0.05, 0.05, 0., 0.05, 0.3, 0.15], # Technical orientation
    [0, 0.1, 0.05, 0.05, 0.3, 0.4, 0, 0.1], # Art orientation
]
weights = [0.2, 0.7, 0.1]

n_schools = 200
n_students_per_school = 30
X = []
y_true = []
for i in range(n_schools):
    school_type = generator.choice(np.arange(3), p=weights)
    y_true.append(school_type)
    # We simulate each graduate as going into one profession and
    # count them in a vector
    school_graduates = np.zeros(len(columns), dtype="int")
    for j in range(n_students_per_school):
        profession = generator.choice(np.arange(len(columns)), p=true_prob[school_type])
        school_graduates[profession] += 1
    X.append(school_graduates)
# X will be a matrix of how many students went into which direction this year
# each school is a row, and each column is a profession
X = np.stack(X)
y_true = np.array(y_true)
```

### Recovering clusters in the data

Let us assume that we don't know what clusters there are in the data (we don't know about art, academic and technical orientation),
and we would like the data to tell us a) what kinds of clusters there are b) how prevalent they are c) which school is which of these orientations.

In order to achieve this we can use a Dirichlet Multinomial Mixture model, which is a Bayesian generative model, that assumes that each of the observations (schools) is generated from one of K Dirichlet-Multinomial distributions.
noloox contains a Gibbs-Sampling implementation of this model in JAX:

```python
from noloox.mixture import DirichletMultinomialMixture

model = DirichletMultinomialMixture(n_components=3, random_state=42)
# We get predictions of the labels for each school
y_pred = model.fit_predict(X)
```

### Cluster interpretation

We can check how well our model's clstering aligns with the true labels using the Adjusted Rand Score:

```python
from sklearn.metrics import adjusted_rand_score

print(adjusted_rand_score(y_true, y_pred))
```

```
1.0
```

We get a value of 1.0, meaning that our model fits the data perfectly and recovers the true clusters.
This of course doesn't tell us which cluster is which category, so we will have to see for ourselves, which cluster predicts what probability of going into each profession.
We can do this by examining the model's parameters:

```python
import plotly.express as px

probs = (model.components_.T / model.components_.sum(axis=1)).T
px.imshow(probs, x=columns, y=[f"Cluster {i}" for i in range(3)])
```

<figure>
  <iframe src="../images/count_clusters.html", title="School clusters", style="height:500px;width:1050px;padding:0px;border:none;"></iframe>
  <figcaption> Probabilities for each profession in each of the discovered clusters. </figcaption>
</figure>

We can clearly see that cluster 0 corresponds to the art-school category, cluster 1 is the more academically oriented schools, while cluster 2 contains schools with a technical orientation.
The model recovers these probabilities very faithfully.

To see how likely each cluster is according to the model, we can plot the `weights_` attribute:
```python
fig = px.bar(x=model.weights_, y=[f"Cluster {i}" for i in range(3)])
fig = fig.update_layout(xaxis_title="probability", yaxis_title="")
fig.show()
```

<figure>
  <iframe src="../images/count_cluster_weights.html", title="School cluster weights", style="height:500px;width:1050px;padding:0px;border:none;"></iframe>
  <figcaption> Cluster probabilities. </figcaption>
</figure>

We can see that the model recovers the weights of the different school types roughly correctly, but slightly underestimates the probability of art schools, and overestimates the probability of technical schools.
This "rich-get-richer" behaviour is typical of this model, and can either be mitigate by collecting more data, or adjusting its priors.
See the documentation on priors [here](DirichletMultinomialMixture.md).
