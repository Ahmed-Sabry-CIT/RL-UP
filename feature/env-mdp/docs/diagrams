# Ahmed Sabry — Assigned Diagrams (Mermaid)

## 1. UML Class Diagram — JointFSDAEnv
```mermaid
classDiagram
    class ParallelEnv {
        <<PettingZoo>>
        +possible_agents: list
        +agents: list
        +reset(seed, options)
        +step(actions)
    }
    class JointFSDAEnv {
        -X_train, y_train, X_val, y_val: ndarray
        -F: int  -K: int  -T: int
        -mask: bool[F]
        -t: int  -stale: int  -best_f1: float
        -n_synth: int
        -redundancy: float[F]
        -train_cluster, val_cluster: int[]
        -alpha, beta, gamma, delta: float
        +observation_space(agent) Box
        +action_space(agent) MultiBinary|Dict
        +state() ndarray
        +reset(seed, options) obs, infos
        +step(actions) obs, rewards, terminated, truncated, infos
        -_parse_actions(actions)
        -_augment(X_sub, mask, method, ratio, cluster)
        -_fit_eval(mask, synth) metrics
        -_distortion(synth, X_sub, cluster) D_syn
        -_cluster_stats(pred, proba)
        -_fs_obs() / _da_obs()
    }
    class DefaultAugmenter {
        +__call__(method, X_cluster, X_all, y_all, n_new, rng)
    }
    class ImportanceProvider {
        <<callable - TreeSHAP hook>>
        +__call__(clf, X_sub, y, mask)
    }
    class ClassifierFactory {
        <<callable>>
        +__call__() LGBM/XGB classifier
    }
    ParallelEnv <|-- JointFSDAEnv
    JointFSDAEnv o-- DefaultAugmenter : augmenter
    JointFSDAEnv o-- ImportanceProvider : importance_provider
    JointFSDAEnv o-- ClassifierFactory : classifier_factory
```

## 2. UML Sequence Diagram — agent–env–classifier loop (one step)
```mermaid
sequenceDiagram
    participant FS as π_FS (fs_agent)
    participant DA as π_DA (da_agent)
    participant E as JointFSDAEnv
    participant C as LightGBM/XGBoost
    participant X as Importance (TreeSHAP) hook
    FS->>E: a_FS (binary mask)
    DA->>E: a_DA (method, ratio, cluster)
    E->>E: _parse_actions (clip ratio, repair empty mask)
    E->>E: project X_train onto mask
    E->>E: _augment in subspace (target cluster)
    E->>C: fit(train ∪ synthetic)
    C-->>E: fitted model
    E->>C: predict_proba(X_val[:, mask])
    C-->>E: proba → Macro-F1, G-Mean, PR-AUC
    E->>X: importance(clf, X_sub, y, mask)
    X-->>E: global importance vector
    E->>E: _distortion → D_syn; compute R_t
    E-->>FS: obs_FS, R_t, term/trunc, info
    E-->>DA: obs_DA, R_t, term/trunc, info
```

## 3. MDP State-Transition & Reward Architecture
```mermaid
flowchart LR
    subgraph S["State s_t (critic) = fs_obs ⊕ da_obs"]
        direction TB
        s1["mask m_t (F)"]
        s2["global importance (F)"]
        s3["redundancy (F)"]
        s4["cluster stats: miss-rate, density, uncertainty (3K)"]
        s5["minority ratio, synth fraction, t/T, F1, G-Mean"]
    end
    S -->|o_FS| AFS["π_FS → a_FS ∈ {0,1}^F"]
    S -->|o_DA| ADA["π_DA → a_DA = (method, ratio, cluster)"]
    AFS --> T["Transition: project subspace → synthesise → retrain classifier → evaluate on validation"]
    ADA --> T
    T --> S2["s_{t+1}"]
    T --> R["R_t (shared)"]
    subgraph RW["R_t = α·ΔF1 + β·GMean − γ·F_sel/F_tot − δ·D_syn"]
        direction TB
        r1["+ α·ΔF1_macro (improvement)"]
        r2["+ β·G-Mean (balanced recall)"]
        r3["− γ·F_sel/F_tot (feature cost)"]
        r4["− δ·D_syn (distribution distortion)"]
    end
    R --- RW
```

## 4. Episode Lifecycle Flowchart
```mermaid
flowchart TD
    A([reset seed]) --> B["Rollback: mask = all ones, synthetic data = ∅, t = 0, stale = 0"]
    B --> C["Fit baseline classifier on full features → baseline F1, obs_0"]
    C --> D{"agents non-empty?"}
    D -- yes --> E["Receive a_FS, a_DA"]
    E --> F{"mask empty?"}
    F -- yes --> G["Force most-important feature (forced_feature)"]
    F -- no --> H
    G --> H["Project subspace, synthesise (invalid cluster → no-op)"]
    H --> I["Retrain classifier, evaluate on validation"]
    I --> J["Compute R_t, t = t+1, update best F1 / stale"]
    J --> K{"stale ≥ patience?"}
    K -- yes --> L([terminated = True])
    K -- no --> M{"t ≥ T?"}
    M -- yes --> N([truncated = True])
    M -- no --> O["Return obs_t, R_t"] --> D
    L --> P["agents = []"]
    N --> P
    P --> A
```
