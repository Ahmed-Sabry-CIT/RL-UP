# RL Project — Joint Feature Selection & Adaptive Data Augmentation

An explainability-guided multi-agent RL framework (CTDE) for imbalanced tabular
learning. Six team members build one component each; this repository holds the
**dataset curation and static baseline pipelines** component
(`feature/baselines`).

| Component | Branch | Owner |
|---|---|---|
| Dataset curation + static baselines | `feature/baselines` | Saeed Nadeem Saeed Shaarawy (931230139) |
| Environment & MDP | `feature/env-mdp` | Ahmed Sabry Hamza Heikal |
| TreeSHAP diagnostics | `feature/xai-shap` | Ziad Khaled Mahran Asran Ahmed |
| Agent 1 (feature selection) | `feature/agent1-fs` | Shehab Hossam El-Din Helmy Mohamed |
| Agent 2 (data augmentation) | `feature/agent2-da` | Joseph Wafik Edward Labib |
| CTDE PPO integration | `feature/marl-ppo` | Ziad Mostafa Aly Kamel |

## What this component delivers

Curated, reproducible, stratified splits for healthcare, financial fraud and
network intrusion data, plus static LightGBM/XGBoost baselines scored on
**Macro-F1**, **G-Mean** and **PR-AUC** — the numbers every downstream agent is
measured against. See [docs/BASELINES.md](docs/BASELINES.md) for the full
interface contract and results.

The merged `feature/env-mdp` component (`src/joint_fsda_env/`) turns those splits
into the PettingZoo `ParallelEnv` the agents act in; its frozen space/reward
contract is [docs/INTERFACE_CONTRACT.md](docs/INTERFACE_CONTRACT.md).

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install -e .                             # makes `python -m baselines` importable

python scripts/fetch_data.py                 # download the three benchmarks (~170 MB)
python -m baselines prepare                  # curate splits + freeze the preprocessor
python -m baselines run                      # 3 datasets x {LightGBM, XGBoost}
python -m baselines verify                   # QA gate: PASS/FAIL over every artefact
pytest -q                                    # unit + end-to-end tests
```

Everything lands in `data/processed/<dataset>/` and `results/`.

## Layout

```
configs/baselines.yaml        seeds, split ratios, preprocessing, model params
src/baselines/
  datasets.py                 loaders + validation for the three benchmarks
  splits.py                   stratified, reproducible train/val/test
  preprocessing.py            traditional uncoupled transform (fit on train only)
  models.py                   LightGBM / XGBoost builders, class weights, SMOTE
  metrics.py                  Macro-F1, G-Mean, PR-AUC, threshold tuning, bootstrap CI
  pipeline.py                 prepare / run / report / verify
  cli.py                      `python -m baselines`
src/joint_fsda_env/
  env.py                      JointFSDAEnv: spaces, step/reset, composite reward (feature/env-mdp)
  augment.py                  SMOTE / ADASYN / noise augmenters (pluggable hook)
  compat.py                   real Gymnasium + PettingZoo, or offline shims
  data.py                     deterministic synthetic demo splits for tests
scripts/fetch_data.py         dataset download + validation
scripts/calibrate_reward.py   rollout-based reward-weight calibration
tests/                        39 unit and end-to-end tests (27 baselines + 12 environment)
docs/BASELINES.md             documentation, results, data interface contract
docs/INTERFACE_CONTRACT.md    frozen MDP / space contract (jointly with the MARL lead)
docs/mdp_diagrams.md          UML, sequence, reward and episode-lifecycle diagrams
```
