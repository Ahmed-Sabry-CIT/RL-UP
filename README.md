# Explainability-Guided Multi-Agent Reinforcement Learning for Joint Feature Selection and Adaptive Data Augmentation

An integrated, cooperative Multi-Agent Reinforcement Learning (MARL) framework that simultaneously identifies informative feature subsets and synthesizes targeted minority-class samples to maximize predictive performance and model parsimony on high-dimensional, imbalanced tabular datasets.

---

## Overview

In real-world domains such as healthcare, financial fraud detection, and network intrusion detection, classifiers suffer from two coupled bottlenecks: **high feature dimensionality** and **severe class imbalance**. Conventional pipelines treat feature selection and data augmentation as isolated, static steps—ignoring how pruning features shifts sample neighborhoods and how oversampling alters feature correlations.

This repository models joint preprocessing as a unified **Markov Decision Process (MDP)** solved via **Centralized Training with Decentralized Execution (CTDE)** using Multi-Agent Proximal Policy Optimization (MAPPO), guided internally and audited externally by **TreeSHAP** explainability.

---

## Key Architecture & Features

* **Cooperative Two-Agent MARL System (CTDE):**
  * **Feature Selection Agent ($\pi_{FS}$):** Observes binary feature masks, correlation redundancy matrices, and global SHAP importance vectors to prune noisy/redundant attributes into a denoised lower-dimensional subspace.
  * **Data Augmentation Agent ($\pi_{DA}$):** Observes the projected subspace, class error rates, local sample density, and high-uncertainty minority pockets to adaptively select synthesis methods (SMOTE, ADASYN, noise injection) and sampling ratios.
* **Explainability-Guided Policy & Auditing (TreeSHAP):**
  * Supplies global feature importance signals to accelerate $\pi_{FS}$ convergence and local misclassified-sample attributions to target vulnerable decision boundaries for $\pi_{DA}$.
  * Generates pre- and post-optimization SHAP summary and dependence diagnostics to verify decision-boundary improvements.
* **Composite Parsimony-Performance Reward:**
  * Evaluates fast downstream gradient-boosted classifiers (LightGBM / XGBoost) across iterative steps using a composite reward balancing minority classification gains ($\Delta\text{F1}_{\text{macro}}$, $\text{G-Mean}$) against active feature complexity ($\vert{}F_{\text{selected}}\vert{}/\vert{}F_{\text{total}}\vert{}$) and synthetic distribution distortion ($D_{\text{syn}}$).

---

## Tech Stack

* **RL & Multi-Agent Environments:** `PyTorch`, `PettingZoo`, `Gymnasium`
* **Downstream Classifiers & Oversampling:** `LightGBM`, `XGBoost`, `imbalanced-learn`, `scikit-learn`
* **Explainable AI (XAI):** `SHAP` (TreeSHAP)
* **Data Processing & Visualization:** `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`

---

## Target Benchmarks & Expected Outcomes

* **Domains:** Healthcare diagnostics, financial fraud detection, and network intrusion detection.
* **Evaluation Metrics:** $\text{Macro-F1}$, $\text{G-Mean}$, $\text{PR-AUC}$, and active feature reduction ratio (targeting a **30%–60% reduction** in active features with minimal synthetic overhead).

---

## Repository Structure

```text
├── data/               # Raw and stratified train/val/test dataset splits
├── baselines/          # Static feature selection & oversampling baseline pipelines
├── envs/               # Custom Gymnasium / PettingZoo MDP environment & reward logic
├── agents/             # Policy networks for Feature Selection (π_FS) & Data Augmentation (π_DA)
├── xai/                # TreeSHAP global/local attribution extractors & diagnostic plots
├── marl/               # PyTorch CTDE Multi-Agent PPO training & evaluation loops
└── notebooks/          # Exploratory analysis, ablation studies & result visualizations
