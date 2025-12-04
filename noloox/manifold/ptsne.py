from typing import Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from noloox.manifold._ptsne import _ParametricTSNE


class pTSNE(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int = 2,
        alpha: float = 1.0,
        perplexity: float = 30,
        random_state: Optional[int] = None,
        metric="euclidean",
        n_epochs: int = 10,
        use_cuda: bool = False,
        learning_rate: float = 0.01,
        batch_size: int = 500,
    ):
        super().__init__()
        self.n_components = n_components
        self.alpha = alpha
        self.perplexity = perplexity
        self.random_state = random_state
        self.use_cuda = use_cuda
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.metric = metric

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.module_ = _ParametricTSNE(
            self.n_features_in_,
            self.n_components,
            self.perplexity,
            alpha=self.alpha,
            hidden_layer_dims=None,
            seed=self.random_state,
            use_cuda=self.use_cuda,
        )
        self.module_.fit(
            X,
            loss_func="kl",
            pretrain=True,
            epochs=self.n_epochs,
            verbose=False,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            metric=self.metric,
        )
        self.embedding_ = self.transform(X)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.embedding_

    def transform(self, X):
        embeddings = []
        for i in range(0, len(X), self.batch_size):
            batch = torch.from_numpy(X[i : i + self.batch_size])
            with torch.no_grad():
                batch_embeddings = self.module_(batch).detach().cpu().numpy()
                embeddings.extend(batch_embeddings)
        return np.stack(embeddings)
