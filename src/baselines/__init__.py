"""Dataset curation and static baseline pipelines for RL Idea #4.

`feature/baselines` component: curated healthcare / fraud / network splits and
uncoupled LightGBM & XGBoost baselines reported under Macro-F1, G-Mean, PR-AUC.
"""

__version__ = "1.0.0"

from .config import Config, load_config
from .metrics import Metrics, evaluate, g_mean, macro_f1, pr_auc, select_threshold
from .preprocessing import TabularPreprocessor
from .splits import Split, stratified_split

__all__ = [
    "Config",
    "Metrics",
    "Split",
    "TabularPreprocessor",
    "__version__",
    "evaluate",
    "g_mean",
    "load_config",
    "macro_f1",
    "pr_auc",
    "select_threshold",
    "stratified_split",
]
