"""Built-in lightweight augmenters (SMOTE-style, ADASYN-style, Gaussian noise).

Agent 2's developer can replace this by passing any callable with the same
signature as `DefaultAugmenter.__call__` (e.g. a wrapper around imbalanced-learn).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

METHODS = ("none", "smote", "adasyn", "noise")


class DefaultAugmenter:
    def __init__(self, k_neighbors=5, noise_scale=0.05):
        self.k = k_neighbors
        self.noise_scale = noise_scale

    def __call__(self, method, X_cluster, X_all, y_all, n_new, rng):
        """Return synthetic minority rows, shape (n_new, n_selected_features).

        method     : one of METHODS
        X_cluster  : minority rows of the targeted sub-cluster (subspace columns only)
        X_all/y_all: full (un-augmented) training set in the same subspace
        """
        if method == "none" or n_new <= 0 or len(X_cluster) == 0:
            return np.empty((0, X_all.shape[1]), dtype=X_all.dtype)
        if method in ("smote", "adasyn") and len(X_cluster) < 2:
            method = "noise"  # degenerate cluster: neighbours undefined
        if method == "smote":
            return self._interpolate(X_cluster, n_new, rng, weights=None)
        if method == "adasyn":
            return self._interpolate(X_cluster, n_new, rng,
                                     weights=self._hardness(X_cluster, X_all, y_all))
        return self._noise(X_cluster, X_all, n_new, rng)

    def _interpolate(self, Xc, n_new, rng, weights):
        k = min(self.k, len(Xc) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc)
        neigh = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        p = None if weights is None else weights / weights.sum()
        base = rng.choice(len(Xc), size=n_new, p=p)
        pick = neigh[base, rng.integers(0, k, size=n_new)]
        lam = rng.random((n_new, 1))
        return Xc[base] + lam * (Xc[pick] - Xc[base])

    def _hardness(self, Xc, X_all, y_all):
        k = min(self.k, len(X_all) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
        idx = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        r = (y_all[idx] == 0).mean(axis=1)
        return r if r.sum() > 0 else np.ones(len(Xc))

    def _noise(self, Xc, X_all, n_new, rng):
        sigma = self.noise_scale * (X_all.std(axis=0) + 1e-8)
        base = rng.integers(0, len(Xc), size=n_new)
        return Xc[base] + rng.normal(0.0, 1.0, size=(n_new, Xc.shape[1])) * sigma
