import numpy as np
import plotly.express as px
# from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups, load_iris
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from noloox.mixture import DirichletMultinomialMixture, StudentsTMixture

corpus = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes")).data

# encoder = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = encoder.encode(corpus, show_progress_bar=True)

embeddings = np.load("../turftopic/_emb/20news_all-MiniLM.npy")

vectorizer = CountVectorizer(
    min_df=10, max_df=0.1, max_features=4000, stop_words="english"
)
dtm = vectorizer.fit_transform(corpus)

model = DirichletMultinomialMixture(10).fit(dtm)

print(model.weights_)

vocab = vectorizer.get_feature_names_out()
for i, comp in enumerate(model.components_):
    top = np.argsort(-comp)[:10]
    if model.weights_[i] > 0:
        print(i, vocab[top])
