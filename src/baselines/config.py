"""Configuration objects shared by the baseline pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .preprocessing import PreprocessingConfig

DEFAULT_CONFIG_PATH = Path("configs/baselines.yaml")


@dataclass
class Config:
    seed: int = 42
    splits: dict[str, float] = field(
        default_factory=lambda: {"train": 0.6, "val": 0.2, "test": 0.2}
    )
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    imbalance: str = "class_weight"
    bootstrap_samples: int = 200
    n_synthetic: int = 8000
    root: Path = field(default_factory=Path.cwd)

    # resolved lazily so a config can be built without touching the filesystem
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))

    def path(self, name: str) -> Path:
        """Resolve one of the configured directories against the project root."""
        value = getattr(self, name)
        return value if value.is_absolute() else (self.root / value)

    def ensure_dirs(self) -> None:
        for name in ("raw_dir", "processed_dir", "results_dir", "logs_dir"):
            self.path(name).mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, dataset: str) -> Path:
        return self.path("processed_dir") / dataset


def load_config(path: str | Path | None = None) -> Config:
    """Read ``configs/baselines.yaml`` (or ``path``) into a :class:`Config`."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    root = Path.cwd()

    if config_path.exists():
        with open(config_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if config_path.resolve().parent.name == "configs":
            root = config_path.resolve().parent.parent
    elif path is not None:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    paths = data.get("paths", {}) or {}
    config = Config(
        seed=int(data.get("seed", 42)),
        splits=dict(data.get("splits", {"train": 0.6, "val": 0.2, "test": 0.2})),
        preprocessing=PreprocessingConfig.from_dict(data.get("preprocessing")),
        models=dict(data.get("models", {})),
        imbalance=str(data.get("imbalance", "class_weight")),
        bootstrap_samples=int(data.get("bootstrap_samples", 200)),
        n_synthetic=int(data.get("n_synthetic", 8000)),
        root=root,
        raw_dir=Path(paths.get("raw", "data/raw")),
        processed_dir=Path(paths.get("processed", "data/processed")),
        results_dir=Path(paths.get("results", "results")),
        logs_dir=Path(paths.get("logs", "logs")),
    )
    return config
