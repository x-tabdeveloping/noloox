
#### Imports

Let's import the packages and tools we are going to use.

```python
import numpy as np
from plotly.subplots import make_subplots
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from noloox.mixture import DirichletMultinomialMixture
```

#### Data Loading

We are going to use a subset of the 20 Newsgroups dataset from scikit-learn.
We only load the alt.atheism forum data for now so that it doesn't take a long time to run the algorithm.

```python
corpus = fetch_20newsgroups(
    subset="all", remove=("headers", "footers", "quotes"), categories=["alt.atheism"]
).data

```

#### Preprocessing

We will use scikit-learn's CountVectorizer to extract a Bag-of-words matrix over our texts.
We filter for too frequent or infrequent words, and filter out stop-words as well.

```python

vectorizer = CountVectorizer(
    min_df=10, max_df=0.1, max_features=4000, stop_words="english"
)
X = vectorizer.fit_transform(corpus)
```

#### Model fitting

We can now fit a Dirichlet Multinomial Mixture to our data.
I chose to use 5 topics, since the results will be easy to display.

```python
model = DirichletMultinomialMixture(5).fit(X)
```

#### Plotting

We plot the top 10 words for each topic on bar charts using Plotly.

```python
fig = make_subplots(rows=1, cols=5, subplot_titles=[f"Topic {i}" for i in range(5)])
vocab = vectorizer.get_feature_names_out()
for i, comp in enumerate(model.components_):
    top = np.argsort(-comp)[:10]
    fig.add_bar(
        y=vocab[top][::-1],
        x=comp[top][::-1],
        row=1,
        col=i+1,
        orientation="h",
        showlegend=False,
    )
fig.show()
```

<figure>
  <iframe src="../images/topics_bar.html", title="Top words for each topic", style="height:800px;width:1050px;padding:0px;border:none;"></iframe>
  <figcaption> Top words in each topic learned by our model. </figcaption>
</figure>
