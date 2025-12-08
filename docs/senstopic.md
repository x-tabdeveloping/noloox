# Topic discovery by factoring transformer embeddings

In older topic models, one would typically cluster or factorize bag-of-words matrices using LDA or NMF. 

!!! info
    See [Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html) in the scikit-learn docs or [Topic modelling for short texts](short_topic_modelling.md) for more information on BoW topic modelling.

In contrast, modern topic models, like BERTopic cluster embeddings from sentence-transformers.
While clustering them is easy, factorizing these embeddings in a nonnegative topic-space is not trivial, as the embeddings themselves are unbounded, and can take on positive or negative values.

In this tutorial we will look at how you can achieve this using [Semi-Nonnegative Matrix Factorization](SNMF.md).

!!! tip
    This model is actually called [SensTopic](https://x-tabdeveloping.github.io/turftopic/SensTopic/), and is implemented in the [Turftopic Python library](https://x-tabdeveloping.github.io/turftopic/) with much more complete functionality.
    If you intend to use this model in practice, you should probably use that implementation, they are based on the same SNMF model.
    This tutorial is strictly here to demonstrate how you can produce positive factors over unbounded data using SNMF.

### Data loading

We will use a subset of the 20 Newsgroups dataset from scikit-learn for this tutorial:

```python
from sklearn.datasets import fetch_20newsgroups

corpus = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
    categories=[
      "comp.graphics",
      "comp.os.ms-windows.misc",
      "comp.sys.ibm.pc.hardware",
      "comp.sys.mac.hardware",
      "comp.windows.x"
    ]
).data
```

### Term extraction

In order for us to be able to estimate keyword importance for topics, we will need to extract all terms in the corpus.
We will do this by getting the vocabulary of a fitted CountVectorizer.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=10)
vectorizer.fit(corpus)
vocab = vectorizer.get_feature_names_out()
```

### Producing transformer embeddings

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(corpus, show_progress_bar=True)
vocab_embeddings = encoder.encode(vocab, show_progress_bar=True)
```

```python
from noloox.decomposition import SNMF

model = SNMF(n_components=10, sparsity=1.0)
doc_topic_matrix = model.fit_transform(embeddings)

topic_word_matrix = model.transform(vocab_embeddings).T
```

```python
for i, comp in enumerate(topic_word_matrix):
    top = np.argsort(-comp)[:10]
    print(f"Topic {i}:", ", ".join(vocab[top]))

```

| Topic ID | Top Words |
| - | - |
| Topic 0 | modem, modems, connecting, telnet, ports, port, connect, ethernet, connects, connection                     | 
| Topic 1 | processors, processor, cpus, cpu, performance, benchmarks, pentium, cheaper, intel, efficient               | 
| Topic 2 | vga, monitors, monitor, 640x480, displays, resolution, resolutions, lcd, 1280x1024, screen                  | 
| Topic 3 | uh, um, so, em, yeah, er, ah, but, oh, and                                                                  | 
| Topic 4 | windows, win3, os, microsoft, openwin, win31, openwindows, netware, ms, executables                         | 
| Topic 5 | printing, printers, printer, prints, print, laserwriter, printed, laserjet, ink, deskjet                    | 
| Topic 6 | hdd, disk, harddisk, disks, drives, fdisk, seagate, hd, drive, partition                                    | 
| Topic 7 | xcreatewindow, xtwindow, xtrealizewidget, xterminal, xterm, xtpointer, xserver, xservers, xt, xdm           | 
| Topic 8 | bitmaps, colormaps, bitmap, imagemagick, colormap, animation, animations, imagewriter, xputimage, photoshop | 
| Topic 9 | archived, mailed, published, contact, subscription, contacted, archive, publications, incorporated, email   | 

