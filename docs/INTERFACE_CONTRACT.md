# JointFSDAEnv — Interface Contract (freeze by Nov 25 with Ziad Mostafa)

`JointFSDAEnv(ParallelEnv)` · agents: `fs_agent` (π_FS), `da_agent` (π_DA) · binary labels, **1 = minority**.
F = number of features, K = number of minority sub-clusters (default 4), T = horizon (default 10).

| Item | Space | Shape / values |
|---|---|---|
| `fs_agent` obs | `Box[0,1]` | `3F+4` = [mask(F), global importance(F), redundancy(F), t/T, macro-F1, G-Mean, selected-fraction] |
| `da_agent` obs | `Box[0,1]` | `3K+4` = [(miss-rate, density, uncertainty)×K, minority ratio, synth fraction, t/T, macro-F1] |
| `state()` (centralised critic) | `Box[0,1]` | `3F+3K+8` = concat(fs_obs, da_obs) |
| `fs_agent` action | `MultiBinary(F)` | 1 = keep feature. Empty mask → most-important feature forced (`forced_feature` flag) |
| `da_agent` action | `Dict` | `method: Discrete(4)` {none, smote, adasyn, noise}; `ratio: Box[0,1](1,)` = fraction of the class-imbalance gap to close; `cluster: Discrete(K)` target minority sub-cluster |

**Step order:** FS mask is applied first → DA synthesises inside the projected subspace → classifier retrained on (train ∪ synthetic) → metrics on **validation only** (test never touched) → shared reward to both agents.

**Reward:** `R_t = α·ΔF1_macro + β·GMean − γ·(F_sel/F_tot) − δ·D_syn`, defaults `α=1, β=0.5, γ=0.5, δ=1.0` (from `scripts/calibrate_reward.py`).
`D_syn` = mean-over-selected-features KS distance between synthetic rows and the real rows of the targeted cluster, scaled by min(1, n_synth / n_cluster_real).

**Episode:** `reset()` restores full mask, no synthetic data (rollback) and fits the baseline. Truncates at T steps; terminates after `patience` steps without macro-F1 improvement. When done, `env.agents == []`.

**`info[agent]` keys:** macro_f1, gmean, pr_auc, importance, cluster_stats, d_f1, feat_frac, d_syn, reward, n_features, feature_reduction, n_synth, method, ratio, cluster, t, forced_feature, invalid_cluster.

## Plug-in hooks
- `importance_provider(clf, X_sub, y, mask) -> ndarray[mask.sum()]` — **Ziad Khaled** replaces the placeholder with global mean-|SHAP|.
- `augmenter(method, X_cluster, X_all, y_all, n_new, rng) -> ndarray` — **Joseph** can wrap imbalanced-learn SMOTE/ADASYN.
- `classifier_factory()` — defaults to LightGBM if installed, else sklearn HistGradientBoosting.

Sub-clusters are computed once on the full feature space (stable IDs across steps, sorted by size). Input data must be finite, scaled and split upstream (Saeed's pipeline).

## Implementation

| Piece | Location |
|---|---|
| `JointFSDAEnv` (spaces, `reset`/`step`, reward, `D_syn`) | `src/joint_fsda_env/env.py` |
| Augmenters (`none`, `smote`, `adasyn`, `noise`) | `src/joint_fsda_env/augment.py` |
| Real Gymnasium/PettingZoo or offline shims | `src/joint_fsda_env/compat.py` |
| Deterministic demo splits (`make_demo_data`) for offline tests | `src/joint_fsda_env/data.py` |
| Reward-weight calibration (`alpha/beta/gamma/delta`) | `scripts/calibrate_reward.py` → `results/env_reward_calibration.txt` |
| Tests (12) | `tests/test_env.py` |
| Diagrams | `docs/mdp_diagrams.md` |

The environment passes the official `pettingzoo.test.parallel_api_test`. Demo data is a
synthetic stand-in: real runs must pass Saeed's curated, scaled splits from
`data/processed/<dataset>/` (`data/processed/<dataset>/schema.json` documents them).
