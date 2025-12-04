"""Implementation from https://github.com/einbandi/ptsne-pytorch/blob/master/ptsne/ptsne.py"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import root_scalar
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import trange

EPS = np.finfo(float).eps
LOG_2 = np.log(2)


def entropy(d, beta):
    x = -d * beta
    y = np.exp(x)
    ysum = y.sum()
    if np.isclose(ysum, 0):
        return -1.0
    else:
        factor = -1 / max((LOG_2 * ysum), EPS)
        return factor * ((y * x) - (y * np.log(ysum))).sum()


def p_i(d, beta):
    x = -d * beta
    y = np.exp(x)
    ysum = max(y.sum(), EPS)
    return y / ysum


def find_beta(d, perp, upper_bound=1e6):
    if np.isclose(d.sum(), 0):
        return 0
    return root_scalar(
        lambda b: entropy(d, b) - np.log2(perp), bracket=(0.0, upper_bound)
    ).root


def p_ij_sym(x, perp, verbose=False, metric="euclidean"):
    num_pts = x.shape[0]
    k = min(num_pts - 1, int(3 * perp))
    index = NearestNeighbors(n_neighbors=k, metric=metric).fit(x)
    neighbors = np.empty((num_pts, k - 1), dtype=int)
    p_ij = np.empty((num_pts, k - 1))
    for i in trange(len(x), desc="Calculating betas"):
        xi = x[i]
        if verbose:
            print(
                "Calculating probabilities: {cur}/{tot}".format(cur=i + 1, tot=num_pts),
                end="\r",
            )
        dists, nn = index.kneighbors([xi], k, return_distance=True)
        beta = find_beta(dists[0, 1:], perp)
        neighbors[i] = nn[0, 1:]
        p_ij[i] = p_i(dists[0, 1:], beta)
    row_indices = np.repeat(np.arange(num_pts), k - 1)
    p = csr_matrix((p_ij.ravel(), (row_indices, neighbors.ravel())))
    return 0.5 * (p + p.transpose())


def dist_mat_squared(x):
    batch_size = x.shape[0]
    expanded = x.unsqueeze(1)
    tiled = torch.repeat_interleave(expanded, batch_size, dim=1)
    diffs = tiled - tiled.transpose(0, 1)
    sum_act = torch.sum(torch.pow(diffs, 2), axis=2)
    return sum_act


def norm_sym(x):
    x.fill_diagonal_(0.0)
    norm_facs = x.sum(axis=0, keepdim=True)
    x = x / norm_facs
    return 0.5 * (x + x.t())


def q_ij(x, alpha):
    dists = dist_mat_squared(x)
    q = torch.pow((1 + dists / alpha), -(alpha + 1) / 2)
    return norm_sym(q)


def kullback_leibler_loss(p, q, eps=1.0e-7):
    eps = torch.tensor(eps, dtype=p.dtype)
    kl_matr = torch.mul(p, torch.log(p + eps) - torch.log(q + eps))
    kl_matr.fill_diagonal_(0.0)
    return torch.sum(kl_matr)


def kullback_leibler_reverse_loss(p, q, eps=1.0e-7):
    return kullback_leibler_loss(q, p, eps)


def jensen_shannon_loss(p, q, eps=1.0e-7):
    m = 0.5 * (p + q)
    return 0.5 * kullback_leibler_loss(p, m, eps) + 0.5 * kullback_leibler_loss(
        q, m, eps
    )


def frobenius_loss(p, q):
    return torch.pow(p - q, 2).sum()


def total_variational_loss(p, q):
    return torch.abs(p - q).sum()


def submatrix(m, indices):
    dim = len(indices)
    indices = np.array(np.meshgrid(indices, indices)).T.reshape(-1, 2).T
    return torch.tensor(m[indices[0], indices[1]].reshape(dim, dim))


class _ParametricTSNE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        perp,
        alpha=1.0,
        hidden_layer_dims=None,
        seed=None,
        use_cuda=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.perplexity = perp
        self.alpha = alpha
        self.use_cuda = use_cuda
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        # If no layers provided, use the same architecture as van der maaten 2009 paper
        if hidden_layer_dims is None:
            hidden_layer_dims = [500, 500, 2000]
        self.layers = nn.ModuleList()
        cur_dim = input_dim
        for hdim in hidden_layer_dims:
            self.layers.append(nn.Linear(cur_dim, hdim))
            cur_dim = hdim
        self.layers.append(nn.Linear(cur_dim, output_dim))
        if self.alpha == "learn":
            self.alpha = nn.Parameter(torch.tensor(1.0))
        if self.use_cuda:
            self.cuda()

    def forward(self, x):
        for layer in self.layers[:-1]:
            # x = torch.sigmoid(layer(x))
            x = F.softplus(layer(x))
        out = self.layers[-1](x)
        return out

    def pretrain(
        self,
        training_data,
        epochs=10,
        verbose=False,
        batch_size=500,
        learning_rate=0.01,
    ):
        if type(training_data) is torch.Tensor:
            pca = torch.tensor(
                PCA(n_components=2).fit_transform(training_data.detach().cpu().numpy())
            )
        else:
            pca = torch.tensor(PCA(n_components=2).fit_transform(training_data))
        dataset = torch.utils.data.TensorDataset(training_data, pca)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for epoch in trange(epochs, desc="Pretraining epochs"):
            running_loss = 0
            for batch, data in enumerate(dataloader):
                features, targets = data
                if self.use_cuda:
                    features = features.cuda()
                    targets = targets.cuda()
                optim.zero_grad()
                loss = criterion(self(features), targets)
                loss.backward()
                optim.step()
                running_loss += loss.item()

    def fit(
        self,
        training_data,
        loss_func="kl",
        p_ij=None,
        pretrain=False,
        epochs=10,
        verbose=False,
        optimizer=torch.optim.Adam,
        batch_size=500,
        learning_rate=0.01,
        metric="euclidean",
    ):
        assert (
            training_data.shape[1] == self.input_dim
        ), "Input training data must be same shape as training `num_inputs`"
        self.p_ij = p_ij
        self._epochs = epochs
        training_data = torch.tensor(training_data)
        if pretrain:
            self.pretrain(
                training_data, epochs=5, verbose=verbose, batch_size=batch_size
            )
        if self.p_ij is None:
            self.p_ij = p_ij_sym(
                training_data.detach().cpu().numpy(),
                self.perplexity,
                verbose=verbose,
                metric=metric,
            ).toarray()
        dataset = torch.utils.data.TensorDataset(
            training_data, torch.arange(training_data.shape[0])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optim = optimizer(self.parameters(), lr=learning_rate)
        loss_func = {
            "kl": kullback_leibler_loss,
            "kl_rev": kullback_leibler_reverse_loss,
            "js": jensen_shannon_loss,
            "frob": frobenius_loss,
            #'bat': bhattacharyya_loss,
            "tot": total_variational_loss,
        }[loss_func]
        for epoch in trange(epochs, desc="Training epochs"):
            running_loss = 0
            for batch, data in enumerate(dataloader):
                features, indices = data
                p = submatrix(self.p_ij, indices.numpy())
                p = p / p.sum()
                if epoch < 10:
                    # exaggeration test
                    exaggeration = 10.0
                    p *= exaggeration
                if self.use_cuda:
                    features = features.cuda()
                    p = p.cuda()
                optim.zero_grad()
                q = q_ij(self(features), self.alpha)
                q = q / q.sum()
                loss = loss_func(p, q)
                if epoch < 10:
                    # exaggeration tets
                    loss = loss / exaggeration - np.log(exaggeration)
                loss.backward()
                optim.step()
                running_loss += loss.item()
