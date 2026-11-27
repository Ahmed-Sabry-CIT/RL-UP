"""JointFSDAEnv — cooperative two-agent PettingZoo ParallelEnv for joint feature
selection (pi_FS) and adaptive data augmentation (pi_DA) on imbalanced binary data.

Reward (shared by both agents, CTDE):
    R_t = alpha*dF1_macro + beta*GMean - gamma*(F_selected/F_total) - delta*D_syn
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, f1_score

from .augment import METHODS, DefaultAugmenter
from .compat import ParallelEnv, spaces

FS, DA = "fs_agent", "da_agent"


def default_classifier_factory(seed=0):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=60, learning_rate=0.1, num_leaves=15,
                              importance_type="gain", n_jobs=1, verbose=-1,
                              random_state=seed)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=60, random_state=seed)


def default_importance_provider(clf, X_sub, y, mask):
    """Placeholder for TreeSHAP (XAI lead's hook). Must return a non-negative vector
    of length mask.sum(). Falls back to |corr(feature, y)| for models without
    feature_importances_."""
    imp = getattr(clf, "feature_importances_", None)
    if imp is not None and len(imp) == X_sub.shape[1]:
        return np.abs(np.asarray(imp, dtype=float))
    yc = y - y.mean()
    Xc = X_sub - X_sub.mean(axis=0)
    return np.abs(Xc.T @ yc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc) + 1e-12)


def compute_metrics(y_true, pred, proba):
    pos = y_true == 1
    sens = float((pred[pos] == 1).mean()) if pos.any() else 0.0
    spec = float((pred[~pos] == 0).mean()) if (~pos).any() else 0.0
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "gmean": float(np.sqrt(sens * spec)),
        "pr_auc": float(average_precision_score(y_true, proba)) if pos.any() else 0.0,
    }


def _ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic for 1-D arrays."""
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.abs(ca - cb).max())


class JointFSDAEnv(ParallelEnv):
    metadata = {"name": "joint_fsda_v0", "is_parallelizable": True, "render_modes": []}

    def __init__(self, X_train, y_train, X_val, y_val, *, horizon=10, n_subclusters=4,
                 alpha=1.0, beta=0.5, gamma=0.5, delta=1.0, patience=4,
                 classifier_factory=None, augmenter=None, importance_provider=None, seed=0):
        X_train, X_val = np.asarray(X_train, float), np.asarray(X_val, float)
        y_train, y_val = np.asarray(y_train).astype(int), np.asarray(y_val).astype(int)
        for name, arr in (("X_train", X_train), ("X_val", X_val)):
            if arr.ndim != 2 or not np.isfinite(arr).all():
                raise ValueError(f"{name} must be a finite 2-D array (impute/scale upstream)")
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError("train/val feature counts differ")
        if not (set(np.unique(y_train)) == set(np.unique(y_val)) == {0, 1}):
            raise ValueError("binary labels {0,1} required in both splits (1 = minority)")
        if (y_train == 1).sum() >= (y_train == 0).sum():
            raise ValueError("label 1 must be the minority class")

        self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val
        self.F = X_train.shape[1]
        self.K = int(n_subclusters)
        self.T = int(horizon)
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.patience = patience
        self.classifier_factory = classifier_factory or (lambda: default_classifier_factory(seed))
        self.augmenter = augmenter or DefaultAugmenter()
        self.importance_provider = importance_provider or default_importance_provider
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        self.possible_agents = [FS, DA]
        self.agents = []

        # ---- static precomputation (redundancy, minority sub-clusters) ----
        corr = np.abs(np.nan_to_num(np.corrcoef(X_train, rowvar=False)))
        np.fill_diagonal(corr, 0.0)
        self.redundancy = (corr.sum(axis=1) / max(self.F - 1, 1)).astype(np.float32)
        self._build_clusters()

        # ---- spaces (the frozen interface contract; see docs/INTERFACE_CONTRACT.md) ----
        self.fs_obs_dim = 3 * self.F + 4
        self.da_obs_dim = 3 * self.K + 4
        self.state_dim = self.fs_obs_dim + self.da_obs_dim
        self._obs_spaces = {
            FS: spaces.Box(0.0, 1.0, shape=(self.fs_obs_dim,), dtype=np.float32),
            DA: spaces.Box(0.0, 1.0, shape=(self.da_obs_dim,), dtype=np.float32)}
        self._act_spaces = {
            FS: spaces.MultiBinary(self.F),
            DA: spaces.Dict({"method": spaces.Discrete(len(METHODS)),
                             "ratio": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                             "cluster": spaces.Discrete(self.K)})}
        self.state_space = spaces.Box(0.0, 1.0, shape=(self.state_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ API
    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self._act_spaces[agent]

    def state(self):
        return np.concatenate([self._fs_obs(), self._da_obs()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self.agents = list(self.possible_agents)
        self.t, self.stale, self.best_f1 = 0, 0, -np.inf
        self.mask = np.ones(self.F, dtype=bool)             # rollback: all features active,
        self.n_synth, self.minority_ratio = 0, self._minority_ratio(0)  # no synthetic data
        self.last = self._fit_eval(self.mask, np.empty((0, self.F)))   # baseline on full data
        self.prev_f1 = self.baseline_f1 = self.last["macro_f1"]
        self.best_f1 = self.prev_f1
        self.last_terms = {"d_f1": 0.0, "gmean": self.last["gmean"],
                           "feat_frac": 1.0, "d_syn": 0.0}
        infos = {a: {"baseline_f1": self.baseline_f1} for a in self.agents}
        return self._observations(), infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}
        mask, method, ratio, cluster, flags = self._parse_actions(actions)

        # --- Agent 1 acts first: project onto the selected subspace
        X_sub = self.X_train[:, mask]
        # --- Agent 2 then synthesises inside that subspace (the "handshake")
        synth, n_new = self._augment(X_sub, mask, method, ratio, cluster, flags)

        m = self._fit_eval(mask, synth, X_sub=X_sub)
        d_syn = self._distortion(synth, X_sub, cluster)
        terms = {"d_f1": m["macro_f1"] - self.prev_f1, "gmean": m["gmean"],
                 "feat_frac": float(mask.sum() / self.F), "d_syn": d_syn}
        reward = (self.alpha * terms["d_f1"] + self.beta * terms["gmean"]
                  - self.gamma * terms["feat_frac"] - self.delta * terms["d_syn"])

        self.t += 1
        self.mask, self.last, self.last_terms = mask, m, terms
        self.n_synth = len(synth)
        self.minority_ratio = self._minority_ratio(self.n_synth)
        self.prev_f1 = m["macro_f1"]
        if m["macro_f1"] > self.best_f1 + 1e-4:
            self.best_f1, self.stale = m["macro_f1"], 0
        else:
            self.stale += 1

        terminated = self.stale >= self.patience
        truncated = self.t >= self.T and not terminated
        obs = self._observations()
        info = {**m, **terms, "reward": float(reward), "n_features": int(mask.sum()),
                "feature_reduction": 1.0 - terms["feat_frac"], "n_synth": self.n_synth,
                "method": METHODS[method], "ratio": ratio, "cluster": cluster,
                "t": self.t, **flags}
        rewards = {a: float(reward) for a in self.agents}     # fully cooperative
        terms_d = {a: terminated for a in self.agents}
        trunc_d = {a: truncated for a in self.agents}
        infos = {a: dict(info) for a in self.agents}
        if terminated or truncated:
            self.agents = []                                  # PettingZoo convention
        return obs, rewards, terms_d, trunc_d, infos

    # ------------------------------------------------------------ internals
    def _parse_actions(self, actions):
        for a in (FS, DA):
            if a not in actions:
                raise KeyError(f"missing action for '{a}'")
        flags = {"forced_feature": False, "invalid_cluster": False}
        mask = np.asarray(actions[FS]).astype(bool).reshape(-1)
        if mask.shape != (self.F,):
            raise ValueError(f"fs action must have shape ({self.F},)")
        if not mask.any():                    # edge case: empty mask -> keep most important feature
            mask = mask.copy()
            mask[int(np.argmax(self._importance_full()))] = True
            flags["forced_feature"] = True
        da = actions[DA]
        method = int(da["method"])
        if not 0 <= method < len(METHODS):
            raise ValueError("invalid augmentation method index")
        ratio = float(np.clip(np.asarray(da["ratio"], float).reshape(-1)[0], 0.0, 1.0))
        cluster = int(da["cluster"])
        if not 0 <= cluster < self.K:
            raise ValueError("invalid cluster index")
        return mask, method, ratio, cluster, flags

    def _augment(self, X_sub, mask, method, ratio, cluster, flags):
        gap = int((self.y_train == 0).sum() - (self.y_train == 1).sum())
        n_new = int(round(ratio * gap))        # ratio = fraction of the imbalance gap to close
        if cluster >= self.n_eff:
            flags["invalid_cluster"] = True    # edge case: cluster id beyond what exists
            return np.empty((0, X_sub.shape[1])), 0
        rows = (self.y_train == 1) & (self.train_cluster == cluster)
        synth = self.augmenter(METHODS[method], X_sub[rows], X_sub, self.y_train, n_new, self.rng)
        return np.asarray(synth, float), n_new

    def _fit_eval(self, mask, synth, X_sub=None):
        X_sub = self.X_train[:, mask] if X_sub is None else X_sub
        Xa = np.vstack([X_sub, synth]) if len(synth) else X_sub
        ya = np.concatenate([self.y_train, np.ones(len(synth), int)])
        clf = self.classifier_factory()
        clf.fit(Xa, ya)
        Xv = self.X_val[:, mask]
        proba = clf.predict_proba(Xv)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = compute_metrics(self.y_val, pred, proba)
        imp = np.zeros(self.F)
        imp[mask] = np.asarray(self.importance_provider(clf, X_sub, self.y_train, mask), float)
        s = imp.sum()
        imp = imp / s if s > 0 else mask / mask.sum()
        m["importance"] = imp.astype(np.float32)
        m["cluster_stats"] = self._cluster_stats(pred, proba)
        return m

    def _distortion(self, synth, X_sub, cluster):
        """D_syn: KS distance between synthetic rows and the real rows of the targeted
        cluster (mean over selected features), scaled by synthetic volume vs minority size."""
        if len(synth) == 0:
            return 0.0
        real = X_sub[(self.y_train == 1) & (self.train_cluster == cluster)]
        ks = np.mean([_ks_stat(synth[:, j], real[:, j]) for j in range(X_sub.shape[1])])
        return float(ks * min(1.0, len(synth) / max(len(real), 1)))

    # --- sub-clusters (computed once on the full feature space => stable ids) ---
    def _build_clusters(self):
        mt, mv = self.y_train == 1, self.y_val == 1
        self.n_eff = int(max(1, min(self.K, mt.sum() // 2)))
        self.train_cluster = np.full(len(self.y_train), -1)
        self.val_cluster = np.full(len(self.y_val), -1)
        if self.n_eff == 1:
            self.train_cluster[mt], self.val_cluster[mv] = 0, 0
        else:
            km = KMeans(self.n_eff, n_init=5, random_state=self._seed).fit(self.X_train[mt])
            order = np.argsort(-np.bincount(km.labels_, minlength=self.n_eff))  # size desc
            remap = np.empty(self.n_eff, int)
            remap[order] = np.arange(self.n_eff)
            self.train_cluster[mt] = remap[km.labels_]
            self.val_cluster[mv] = remap[km.predict(self.X_val[mv])]
        sizes = np.bincount(self.train_cluster[mt], minlength=self.K)[: self.K]
        self.cluster_share = (sizes / mt.sum()).astype(np.float32)

    def _cluster_stats(self, pred, proba):
        """Per sub-cluster: [local miss rate, density (share of minority), uncertainty]."""
        st = np.zeros((self.K, 3), np.float32)
        for k in range(self.n_eff):
            v = self.val_cluster == k
            st[k, 1] = self.cluster_share[k]
            if v.any():
                st[k, 0] = float((pred[v] != 1).mean())
                st[k, 2] = float((1.0 - np.abs(2.0 * proba[v] - 1.0)).mean())
        return st

    def _minority_ratio(self, n_synth):
        n_min = (self.y_train == 1).sum() + n_synth
        return float(n_min / (len(self.y_train) + n_synth))

    def _importance_full(self):
        return self.last["importance"] if hasattr(self, "last") else np.ones(self.F)

    # --- observations ---
    def _fs_obs(self):
        t = self.t / self.T
        return np.concatenate([self.mask.astype(np.float32), self.last["importance"],
                               self.redundancy,
                               [t, self.last["macro_f1"], self.last["gmean"],
                                self.mask.sum() / self.F]]).astype(np.float32)

    def _da_obs(self):
        synth_frac = min(1.0, self.n_synth / max(1, int((self.y_train == 0).sum()
                                                        - (self.y_train == 1).sum())))
        return np.concatenate([self.last["cluster_stats"].reshape(-1),
                               [self.minority_ratio, synth_frac, self.t / self.T,
                                self.last["macro_f1"]]]).astype(np.float32)

    def _observations(self):
        return {FS: self._fs_obs(), DA: self._da_obs()}
