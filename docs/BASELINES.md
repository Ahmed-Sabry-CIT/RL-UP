# Dataset Curation & Baseline Pipelines

**Component:** `feature/baselines` · **Owner:** Saeed Nadeem Saeed Shaarawy (931230139)
**Schedule:** Days 1–4 (Nov 24–27, 2026) · **Status: complete and verified**

This module is Gate 1 of the project plan: it produces the curated datasets, the
frozen preprocessing contract and the static reference numbers (Macro-F1,
G-Mean, PR-AUC) that every downstream agent — TreeSHAP diagnostics, π_FS, π_DA
and the CTDE PPO pipeline — is built on and compared against.

---

## 1. Delivered against the schedule

| Schedule item | What was delivered |
|---|---|
| Days 1–2: gather healthcare / fraud / network benchmarks, stratified train/val/test | `scripts/fetch_data.py` (download + validation), `src/baselines/datasets.py`, `src/baselines/splits.py` → curated parquet splits + `schema.json` per dataset |
| Day 3: preprocessing scripts + static LightGBM/XGBoost baselines, uncoupled preprocessing | `src/baselines/preprocessing.py`, `src/baselines/models.py`, `src/baselines/pipeline.py`, `python -m baselines run` |
| Day 4: test, debug, verify Macro-F1 / G-Mean / PR-AUC | `tests/` (27 tests), `python -m baselines verify` (97 checks, PASS), results in `results/` |
| Protocol: feature branch, documented module, verified output logs/plots | branch `feature/baselines`, this document, `results/verify_report.txt` (full gate output), `results/baseline_metrics.png` |

---

## 2. Datasets (real benchmark data)

| Domain | Dataset | Rows | Raw → model features | Task | Class support | Imbalance |
|---|---|---:|---:|---|---|---:|
| Healthcare | UCI ML Repo 296 — Diabetes 130-US hospitals (1999–2008) | 101,766 | 47 → 92 | 3-class readmission | NO 54,864 · >30 d 35,545 · <30 d 11,357 | 4.8 : 1 |
| Fraud | ULB Credit Card Fraud Detection (OpenML 1597) | 284,807 | 30 → 30 | binary | normal 284,315 · fraud 492 | 577.9 : 1 |
| Network | NSL-KDD (`KDDTrain+` + `KDDTest+`, pooled) | 148,517 | 41 → 59 | 5-class attack type | Normal 77,054 · DoS 53,387 · Probe 14,077 · R2L 3,880 · U2R 119 | 647.5 : 1 |

Provenance, row counts and class counts are copied into each
`data/processed/<dataset>/schema.json`, and each result JSON repeats
`data_source` (`real` / `synthetic`) and `provenance`, so a table can never be
mistaken for a different dataset or for synthetic smoke-test output.

Re-fetch and re-validate at any time:

```bash
.venv/Scripts/python.exe scripts/fetch_data.py --check
```

---

## 3. Split protocol

* Stratified on the target, `train / val / test = 0.60 / 0.20 / 0.20`,
  `seed = 42` (`configs/baselines.yaml`).
* Every row keeps a **stable positional `row_id`** from the curated pool;
  `assert_disjoint` and `baselines verify` prove the three splits share no row.
* Class coverage is asserted in every split, so the 119-sample U2R class is
  still present in train, validation and test.
* Resulting sizes: healthcare 61,059 / 20,353 / 20,354 · fraud 170,883 / 56,962
  / 56,962 · network 89,109 / 29,704 / 29,704.

---

## 4. Preprocessing — traditional and *uncoupled*

One static transform, fitted on the **training split only** and reused verbatim
for validation, test and downstream agents:

1. drop unusable columns — all-missing, missing fraction above 0.45,
   identifier-like, or more than 100 distinct levels;
2. median / most-frequent imputation, plus `*__is_missing` indicator columns
   when a column had missings in training;
3. standard scaling of numeric features;
4. one-hot encoding of categorical features, grouping levels below 1 %
   frequency (`min_frequency = 0.01`) and ignoring unseen levels at transform
   time.

Dropped columns are recorded per dataset, e.g. healthcare drops `weight`
(96.9 % missing), `A1Cresult` (83.3 %), `max_glu_serum` (94.8 %),
`medical_specialty` (49.1 %) and the free-text ICD-9 codes `diag_1/2/3`
(676–725 levels). Fractional columns are kept as numeric, so one-hot is not the
only encoding and the transform stays fast.

There is **no feedback** from feature selection or augmentation into this
transform — that coupling is exactly what the MARL track adds, and keeping this
stage uncoupled is what makes it a fair baseline.

---

## 5. Models and imbalance handling

| | LightGBM | XGBoost |
|---|---|---|
| Version | 4.7.0 | 2.1.4 |
| Trees / LR | 1000, 0.05 | 1000, 0.05 |
| Leaves / depth | 63 leaves, `min_child_samples` 20 | depth 6, `min_child_weight` 1 |
| Regularisation | `subsample` 0.9, `colsample_bytree` 0.9, `reg_lambda` 1.0 | same |
| Objective | `binary` / `multiclass` | `binary:logistic` / `multi:softprob` |
| Early stopping | 50 rounds on validation log-loss | 50 rounds on validation log-loss |

**Imbalance handling** is traditional: inverse-frequency **balanced sample
weights applied exactly once**, in `fit_model`. (Setting the estimator's own
`class_weight` *and* `sample_weight` squares the weights and pushes the model
far too hard toward the minority class — measured at 0.314 Macro-F1 vs 0.444 on
healthcare before this was fixed.) `--imbalance smote` applies SMOTE to the
training split only; `--imbalance none` disables weighting, for an ablation.

**Decision rule.** Multiclass: argmax of the class probabilities. Binary: the
threshold is tuned on the **validation** split to maximise validation Macro-F1
(ties broken by G-Mean) and then frozen for the test split — the test split is
never used for threshold selection or model selection.

---

## 6. Metric definitions

With `TP_c / FP_c / FN_c` per class `c` and `R_c = TP_c / (TP_c + FN_c)`:

* **Macro-F1** — unweighted mean of per-class F1: `(1/K) Σ_c 2·P_c·R_c/(P_c+R_c)`.
  Treats all classes equally, so ignoring the rare class is punished.
* **G-Mean** — `(Π_c R_c)^(1/K)`, i.e. 0.0 as soon as any class is never
  recovered. For two classes this is exactly `sqrt(TPR · TNR)` (Kubat & Matwin).
* **PR-AUC** — threshold-free average precision, macro one-vs-rest for
  multiclass. This is the metric that stays informative at 0.17 % positives.

Confidence intervals are a **stratified** percentile bootstrap (200 resamples,
each class resampled inside itself so the imbalance ratio is preserved; plain
resampling lets rare classes drop out and skews the interval). Report as
`mean [lo, hi]` at 95 %.

---

## 7. Verified results (real data, test split)

| Dataset | Model | Macro-F1 | G-Mean | PR-AUC | Accuracy | F1 95 % CI |
|---|---|---:|---:|---:|---:|---|
| Healthcare (3-class) | LightGBM | 0.4426 | 0.4167 | 0.4562 | 0.5267 | [0.435, 0.449] |
| Healthcare (3-class) | XGBoost | **0.4454** | **0.4477** | **0.4617** | 0.5127 | [0.439, 0.452] |
| Fraud (binary) | LightGBM | 0.6176 | 0.9483 | 0.9390 | 0.9901 | [0.610, 0.628] |
| Fraud (binary) | XGBoost | **0.6216** | **0.9485** | **0.9410** | 0.9905 | [0.613, 0.632] |
| Network (5-class) | LightGBM | **0.9444** | **0.9642** | 0.9600 | 0.9961 | [0.921, 0.969] |
| Network (5-class) | XGBoost | 0.9430 | 0.9637 | **0.9642** | 0.9958 | [0.920, 0.967] |

Frozen binary thresholds: fraud-LightGBM 0.00152, fraud-XGBoost 0.00435 (both
selected on validation). Full tables: `results/baseline_summary.md` / `.csv`,
plot: `results/baseline_metrics.png`, per-run detail: `results/<dataset>__<model>.json`.

Interpretation for the team:

* Both libraries are within 0.004 Macro-F1 of each other on every dataset, so
  the reference numbers are a property of the data, not of one library. Use
  **LightGBM as the default reference** (fastest fit) and keep XGBoost as the
  cross-check the plan asks for.
* Healthcare is the hard case: 0.44 Macro-F1 with 4.8:1 imbalance is where the
  MARL framework has room to show a gain; fraud and network are already strong
  (0.62 / 0.94), so improvements there must be reported with CIs, not raw
  point differences.
* G-Mean ≈ 0.95 on fraud with Macro-F1 0.62 shows the expected shape of an
  aggressively-thresholded imbalanced model: almost all positives are found,
  at the cost of precision on the fraud class.

---

## 8. Artefacts and interface contract

Everything lives under `data/processed/<dataset>/`:

| Path | Contents |
|---|---|
| `train.parquet`, `val.parquet`, `test.parquet` | columns `row_id`, all raw features, `__target__` (integer label) |
| `schema.json` | `target_names`, `labels` (codes, aligned by position), `task`, `positive_class`, `class_counts`, `imbalance_ratio`, per-split counts/fractions, `n_features_raw`, `n_features_model`, `feature_names` (exact model column order), `dropped_columns`, `seed`, `source`, `provenance`. **This file is tracked in git** (~10 KB in total) so downstream components can read the frozen contract without downloading the raw data; everything else under `data/processed/` is regenerable and ignored. |
| `arrays.npz` | `train_X/val_X/test_X` (float32 model matrices, already preprocessed) and `train_y/val_y/test_y` — ready to feed policy networks |
| `preprocessor.joblib` | the fitted `TabularPreprocessor`; `transform(DataFrame) -> np.ndarray`, `feature_names_out_` |

Downstream usage:

```python
from baselines.pipeline import load_prepared
from baselines.config import load_config

config = load_config()
prepared = load_prepared(config, "fraud")

prepared.splits["train"].X        # raw features (DataFrame)
prepared.splits["train"].y        # integer codes, aligned with prepared.target_names
prepared.preprocessor.transform(...)   # the frozen feature space
prepared.feature_names            # exact column order for π_FS masks
```

Rules the other components can rely on:

* **Label order is fixed** by `schema["target_names"]`; never re-derive it from
  the data (a split may not contain every class after filtering).
* **Never refit the preprocessor** on validation/test data — use
  `preprocessor.joblib` or `arrays.npz`.
* **Feature masks (π_FS)** index `prepared.feature_names`, so a selected subset
  is directly comparable with the `top_features` list in each result JSON.
* **Reward reference**: `ΔF1_macro` in the CTDE reward should be measured
  against the matching static baseline in `results/` for the same split and
  seed (seed 42), not against zero.

For the XAI track, each result JSON already carries `top_features`
(name, raw importance, share), which is a ready sanity-check target for the
TreeSHAP global importance vectors.

---

## 9. Reproducing everything

```bash
# Windows
.venv/Scripts/python.exe scripts/fetch_data.py            # data (idempotent, validates)
.venv/Scripts/python.exe -m baselines prepare --strict     # curated splits + contract
.venv/Scripts/python.exe -m baselines run --strict         # 3 datasets x 2 models
.venv/Scripts/python.exe -m baselines verify               # QA gate, prints PASS/FAIL
.venv/Scripts/python.exe -m pytest -q                      # 27 tests

# Linux/macOS: same commands with .venv/bin/python
```

Useful flags: `--dataset healthcare`, `--model lightgbm`,
`--imbalance {class_weight,smote,none}`, `--strict` (refuse the synthetic
fallback), `--verbose`.

Runtime on a laptop CPU: curation ≈ 8 s, all six baselines ≈ 60 s
(LightGBM 2–8 s per fit, XGBoost 4–16 s), report + verify ≈ 2 s.

---

## 10. Verification performed

* **27 automated tests** — metric definitions (hand-computed Macro-F1, G-Mean =
  `sqrt(TPR·TNR)`, perfect-separation PR-AUC = 1, threshold tuning), split
  integrity (sizes, stratification, reproducibility, zero overlap, rejection of
  impossible ratios/singleton classes), preprocessing (column dropping,
  missing indicators, no train→test refitting, unseen categories, serialisation
  round-trip) and an end-to-end run (curate → fit → evaluate → report → verify).
* **`baselines verify` → 97 checks, PASS**: artefacts exist for all three
  datasets, splits disjoint and correctly sized, all classes present with
  labels in range, metric values finite and within [0, 1], confusion matrices
  square and summing to `n_samples`, binary thresholds in [0, 1], summary
  report present.
* **Three bugs were found and fixed by this verification**, which is why it is
  worth keeping: split overlap was undetectable after per-split re-indexing;
  class weights were applied twice (0.314 → 0.443 Macro-F1 on healthcare);
  XGBoost 2.x rejected `early_stopping_rounds` in `fit()`.

Reproduce the gate at any time with `python -m baselines verify`; it exits
non-zero on failure so it can be wired into CI before the Dec 5 review. The
full output of the last gate run is committed as `results/verify_report.txt`,
so the verified log travels with the repository (the verbose run logs in
`logs/` stay local and are git-ignored).

---

## 11. Known limitations (read before quoting these numbers)

1. **Fraud uses a random stratified split.** The ULB data is time-ordered and
   contains near-duplicate transactions, so a random split lets test rows
   resemble training rows and makes PR-AUC optimistic (0.94). This is
   deliberate: the plan specifies stratified splits, and every agent must be
   compared on *these* splits. A temporal split (by `Time`) is the honest
   variant if the final report has space for a robustness check — it would
   lower all fraud numbers.
2. **NSL-KDD is pooled** (`KDDTrain+` + `KDDTest+`) before splitting, again to
   follow the 60/20/20 protocol; KDD is known to contain duplicate records
   across the official train/test files, so 0.944 Macro-F1 is not comparable to
   papers that test on `KDDTest+` alone.
3. **Healthcare drop-outs:** the free-text ICD-9 codes and three columns with
   83–97 % missing values are dropped, matching common practice. Keeping them
   (better ICD grouping) would likely raise Macro-F1 and is the obvious next
   refinement.
4. **Macro-F1 is threshold-sensitive.** For binary problems the threshold is
   validated on one 20 % split, so its CI is slightly optimistic; the point
   estimate is stable but do not compare thresholds across seeds.
5. **The synthetic fallback** (`--data-source` absent, i.e. no `--strict`) only
   exists so the pipeline runs offline; it is labelled `synthetic` everywhere
   and its numbers are not benchmarks.

---

## 12. Collaboration protocol

* Work stays on `feature/baselines`; PR into `main` only after
  `baselines verify` and `pytest` both pass.
* If a downstream component needs a schema change (new target encoding, extra
  split, different seed), change it **here** and regenerate — do not patch the
  parquet files by hand, or the reference numbers stop matching.
* Regenerating with a different `seed` in `configs/baselines.yaml` invalidates
  every table in `results/`; regenerate them in the same commit.
