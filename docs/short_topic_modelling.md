
# Topic modelling for short texts

[Latent Dirichlet Allocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) is a very popular model for finding topics in text documents.
Crucially, however, LDA is very bad at discovering topics in short documents.
This is a problem when you are trying to model topics in Tweets or forum posts, because this sort of content is typically short-form.

The reason LDA is so bad at this, is because it assumes that every document contains multiple topics, while this is usually not the case with shorter documents.
We can, instead, assume that each document comes from one underlying cluster that determines word probabilities.
This model is called a _Dirichlet-Multinomial Mixture_, and can be used for clustering text, as well as uncovering topics.

In this tutorial we will look at how you can use [DirichletMultinomialMixture](DirichletMultinomialMixture.md) to find topics in short texts.

### Data Loading

We are going to use a subset of the 20 Newsgroups dataset from scikit-learn.
We only load the alt.atheism forum data for now so that it doesn't take a long time to run the algorithm.

```python
from sklearn.datasets import fetch_20newsgroups

corpus = fetch_20newsgroups(
    subset="all", remove=("headers", "footers", "quotes"), categories=["alt.atheism"]
).data

```

### Preprocessing

We will use scikit-learn's [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to extract a Bag-of-words matrix over our texts.
We filter for too frequent or infrequent words, and filter out stop-words as well.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    min_df=10, max_df=0.1, max_features=4000, stop_words="english"
)
X = vectorizer.fit_transform(corpus)
```

### Model fitting

We can now fit a Dirichlet Multinomial Mixture to our data.
I chose to use 5 topics, since the results will be easy to display.

```python
from noloox.mixture import DirichletMultinomialMixture

model = DirichletMultinomialMixture(5).fit(X)
```

### Model Interpretation

We plot the top 10 words for each topic on bar charts using Plotly to understand what topics mean.

```python
from plotly.subplots import make_subplots
import numpy as np

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
