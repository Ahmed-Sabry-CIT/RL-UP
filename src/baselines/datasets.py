"""Benchmark dataset curation for the three target domains.

| Domain     | Dataset                                   | Task        |
|------------|-------------------------------------------|-------------|
| Healthcare | UCI Diabetes 130-US hospitals (1999-2008) | 3-class readmission |
| Fraud      | ULB Credit Card Fraud Detection           | binary fraud        |
| Network    | NSL-KDD                                   | 5-class attack type |

Each loader returns a :class:`RawDataset` with an integer-coded target and a
stable ``target_names`` order.  Every loader first looks for the raw files in
``data/raw``; when they are absent it either raises an actionable error or falls
back to a documented synthetic stand-in (``allow_synthetic=True``), which exists
only so the pipeline can be smoke-tested offline.  Synthetic runs are always
labelled ``source="synthetic"`` in the schema and results so they can never be
mistaken for benchmark numbers.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATASETS = ("healthcare", "fraud", "network")

NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

ATTACK_TO_CATEGORY = {
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "apache2": "DoS", "worm": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

HEALTHCARE_TARGET_NAMES = ["NO", ">30", "<30"]
FRAUD_TARGET_NAMES = ["normal", "fraud"]
NETWORK_TARGET_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

HEALTHCARE_ID_COLUMNS = ["encounter_id", "patient_nbr"]


def _read_csv_sniff(path: Path) -> pd.DataFrame:
    """Read a CSV whose extension may lie about gzip compression.

    Some mirrors serve a plain CSV under a ``.csv.gz`` name (or a proxy
    transparently decodes it), so the gzip magic bytes are inspected instead of
    trusting the suffix.
    """
    compression = None
    if path.suffix == ".gz":
        with open(path, "rb") as handle:
            compression = "gzip" if handle.read(2) == b"\x1f\x8b" else None
    return pd.read_csv(path, compression=compression, low_memory=False)


@dataclass
class RawDataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    target_names: list[str]
    source: str                     # "real" | "synthetic"
    provenance: str
    id_columns: list[str] = field(default_factory=list)
    positive_class: str | None = None
    task: str = "multiclass"        # "binary" | "multiclass"

    @property
    def labels(self) -> list[int]:
        return list(range(len(self.target_names)))

    @property
    def n_classes(self) -> int:
        return len(self.target_names)

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.y.to_numpy(dtype=int), minlength=self.n_classes)
        return {name: int(counts[i]) for i, name in enumerate(self.target_names)}

    def imbalance_ratio(self) -> float:
        """Size of the largest class divided by the size of the smallest."""
        counts = [c for c in self.class_counts().values() if c > 0]
        return float(max(counts) / max(min(counts), 1))


# --------------------------------------------------------------------- healthcare
def _extract_healthcare(raw_dir: Path) -> Path | None:
    csv_path = raw_dir / "diabetic_data.csv"
    if csv_path.exists():
        return csv_path
    for archive in raw_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(archive) as zf:
                member = next(
                    (n for n in zf.namelist() if n.endswith("diabetic_data.csv")), None
                )
                if member is None:
                    continue
                with zf.open(member) as src, open(csv_path, "wb") as dst:
                    dst.write(src.read())
                logger.info("Extracted %s -> %s", member, csv_path)
                return csv_path
        except zipfile.BadZipFile:
            continue
    return None


def load_healthcare(raw_dir: Path) -> RawDataset:
    path = _extract_healthcare(raw_dir)
    if path is None:
        raise FileNotFoundError(
            "Healthcare data missing. Run `python scripts/fetch_data.py --dataset healthcare` "
            f"or place diabetic_data.csv in {raw_dir}."
        )
    frame = pd.read_csv(path, na_values=["?", "None", ""], low_memory=False)
    if "readmitted" not in frame.columns:
        raise ValueError(f"{path} does not look like the UCI diabetes readmission table")

    mapping = {name: i for i, name in enumerate(HEALTHCARE_TARGET_NAMES)}
    unknown = set(frame["readmitted"].dropna().unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Unexpected readmitted values: {sorted(unknown)}")

    y = frame["readmitted"].map(mapping)
    keep = y.notna()
    frame, y = frame.loc[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    ids = [c for c in HEALTHCARE_ID_COLUMNS if c in frame.columns]
    X = frame.drop(columns=ids + ["readmitted"])

    return RawDataset(
        name="healthcare",
        X=X,
        y=y.astype(int),
        target_names=list(HEALTHCARE_TARGET_NAMES),
        source="real",
        provenance="UCI ML Repository 296: Diabetes 130-US hospitals for years 1999-2008",
        id_columns=ids,
        task="multiclass",
    )


# -------------------------------------------------------------------------- fraud
def load_fraud(raw_dir: Path) -> RawDataset:
    candidates = [
        raw_dir / "creditcard.csv",
        raw_dir / "creditcard.csv.gz",
        raw_dir / "phpKo8OWT.csv.gz",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Fraud data missing. Run `python scripts/fetch_data.py --dataset fraud` "
            f"or place creditcard.csv[.gz] in {raw_dir}."
        )

    frame = _read_csv_sniff(path)
    frame.columns = [str(c).strip().strip("'\"") for c in frame.columns]
    target_column = "Class" if "Class" in frame.columns else frame.columns[-1]
    frame = frame.rename(columns={target_column: "Class"})
    # Some mirrors export ARFF-style single-quoted values (e.g. "'0'").
    y = (
        frame["Class"]
        .astype(str)
        .str.strip()
        .str.strip("'\"")
        .astype(int)
    )
    X = frame.drop(columns=["Class"])

    return RawDataset(
        name="fraud",
        X=X,
        y=y,
        target_names=list(FRAUD_TARGET_NAMES),
        source="real",
        provenance="ULB Machine Learning Group: Credit Card Fraud Detection (OpenML 1597)",
        positive_class="fraud",
        task="binary",
    )


# ------------------------------------------------------------------------ network
def load_network(raw_dir: Path) -> RawDataset:
    paths = [raw_dir / "KDDTrain+.txt", raw_dir / "KDDTest+.txt"]
    present = [p for p in paths if p.exists()]
    if not present:
        raise FileNotFoundError(
            "Network data missing. Run `python scripts/fetch_data.py --dataset network` "
            f"or place KDDTrain+.txt / KDDTest+.txt in {raw_dir}."
        )

    frames = []
    for path in present:
        frame = pd.read_csv(path, header=None)
        if frame.shape[1] < len(NSL_KDD_FEATURES) + 2:
            raise ValueError(f"{path} has {frame.shape[1]} columns; expected 43")
        frame = frame.iloc[:, : len(NSL_KDD_FEATURES) + 2]
        frame.columns = NSL_KDD_FEATURES + ["label", "difficulty"]
        frames.append(frame)
        logger.info("Loaded %s (%d rows)", path.name, len(frame))

    frame = pd.concat(frames, ignore_index=True)
    frame["label"] = frame["label"].astype(str).str.strip().str.rstrip(".")
    frame["attack_category"] = frame["label"].map(
        lambda value: "Normal" if value == "normal"
        else ATTACK_TO_CATEGORY.get(value, "UNMAPPED")
    )

    unmapped = sorted(set(frame.loc[frame["attack_category"] == "UNMAPPED", "label"]))
    if unmapped:
        raise ValueError(f"Unmapped NSL-KDD attack labels: {unmapped}")

    mapping = {name: i for i, name in enumerate(NETWORK_TARGET_NAMES)}
    y = frame["attack_category"].map(mapping).astype(int)
    X = frame[NSL_KDD_FEATURES].copy()

    return RawDataset(
        name="network",
        X=X,
        y=y,
        target_names=list(NETWORK_TARGET_NAMES),
        source="real",
        provenance="NSL-KDD (KDDTrain+ and KDDTest+, pooled) -- attack-type taxonomy",
        task="multiclass",
    )


REAL_LOADERS = {
    "healthcare": load_healthcare,
    "fraud": load_fraud,
    "network": load_network,
}


# ---------------------------------------------------------------------- synthetic
def _synthetic_healthcare(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    n = 6000
    y = rng.choice(3, size=n, p=[0.53, 0.36, 0.11])
    signal = np.where(y == 2, 1.4, np.where(y == 1, 0.6, -0.5))
    frame = pd.DataFrame(
        {
            "race": rng.choice(["Caucasian", "AfricanAmerican", "Other"], n, p=[0.75, 0.18, 0.07]),
            "gender": rng.choice(["Female", "Male"], n),
            "age": rng.choice(["[50-60)", "[60-70)", "[70-80)", "[80-90)"], n),
            "time_in_hospital": rng.integers(1, 15, n),
            "num_lab_procedures": rng.normal(43 + 3 * signal, 15, n).round(1),
            "num_procedures": rng.integers(0, 7, n),
            "num_medications": rng.normal(16 + 4 * signal, 8, n).round(1),
            # rates must stay strictly positive for every class
            "number_outpatient": rng.poisson(np.clip(0.4 + 0.2 * signal, 0.01, None), n),
            "number_emergency": rng.poisson(np.clip(0.2 + 0.5 * signal, 0.01, None), n),
            "number_inpatient": rng.poisson(np.clip(0.4 + 1.1 * signal, 0.01, None), n),
            "diag_1": rng.choice(["250.01", "428", "414", "786", "410"], n),
            "insulin": rng.choice(["No", "Steady", "Up", "Down"], n),
            "A1Cresult": rng.choice([">7", ">8", "Norm", "None"], n, p=[0.2, 0.1, 0.15, 0.55]),
        }
    )
    # a deliberately sparse column, like the real table
    sparse_rows = rng.random(n) < 0.9
    frame["A1Cresult"] = frame["A1Cresult"].astype("object")
    frame.loc[sparse_rows, "A1Cresult"] = np.nan
    frame["num_medications"] = frame["num_medications"].abs()
    return frame, y.astype(int)


def _synthetic_fraud(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    n = 8000
    y = (rng.random(n) < 0.017).astype(int)
    frame = pd.DataFrame({"Time": rng.integers(0, 172_792, n)})
    for i in range(1, 11):
        frame[f"V{i}"] = rng.normal(0, 1.0, n) + y * rng.normal(0.6, 1.0)
    frame["Amount"] = np.abs(rng.lognormal(3.0, 1.2, n)).round(2)
    return frame, y


def _synthetic_network(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    n = 8000
    y = rng.choice(5, size=n, p=[0.47, 0.36, 0.12, 0.045, 0.005])
    signal = np.where(y == 0, 0.0, 1.2)
    frame = pd.DataFrame(
        {
            "duration": rng.integers(0, 4000, n),
            "protocol_type": rng.choice(["tcp", "udp", "icmp"], n, p=[0.7, 0.2, 0.1]),
            "service": rng.choice(["http", "private", "ftp_data", "other"], n, p=[0.4, 0.35, 0.15, 0.1]),
            "flag": rng.choice(["SF", "S0", "REJ"], n, p=[0.6, 0.3, 0.1]),
            "src_bytes": np.abs(rng.lognormal(6 + signal, 2, n)).astype(int),
            "dst_bytes": np.abs(rng.lognormal(7 - signal, 2, n)).astype(int),
            "count": rng.integers(1, 300, n),
            "srv_count": rng.integers(1, 300, n),
            "serror_rate": np.round(rng.beta(1 + 3 * signal, 6, n), 2),
            "same_srv_rate": np.round(rng.beta(5 - 2 * signal, 2, n), 2),
            "dst_host_count": rng.integers(1, 255, n),
            "dst_host_srv_count": rng.integers(1, 255, n),
        }
    )
    return frame, y.astype(int)


SYNTHETIC_BUILDERS = {
    "healthcare": (_synthetic_healthcare, HEALTHCARE_TARGET_NAMES),
    "fraud": (_synthetic_fraud, FRAUD_TARGET_NAMES),
    "network": (_synthetic_network, NETWORK_TARGET_NAMES),
}


def make_synthetic(name: str, seed: int = 42) -> RawDataset:
    builder, target_names = SYNTHETIC_BUILDERS[name]
    rng = np.random.default_rng(seed)
    X, y = builder(rng)
    return RawDataset(
        name=name,
        X=X,
        y=pd.Series(y),
        target_names=list(target_names),
        source="synthetic",
        provenance=(
            "SYNTHETIC stand-in with the real dataset's schema and imbalance "
            "profile -- for offline smoke tests only, NOT benchmark numbers."
        ),
        task="binary" if len(target_names) == 2 else "multiclass",
        positive_class=target_names[1] if len(target_names) == 2 else None,
    )


def load(
    name: str,
    raw_dir: str | Path,
    allow_synthetic: bool = True,
    seed: int = 42,
) -> RawDataset:
    """Load ``name`` from raw files, falling back to synthetic when allowed."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}")
    raw_dir = Path(raw_dir)
    try:
        return REAL_LOADERS[name](raw_dir)
    except FileNotFoundError as exc:
        if not allow_synthetic:
            raise
        logger.warning(
            "%s -- falling back to the SYNTHETIC stand-in (results are not benchmarks)",
            exc,
        )
        return make_synthetic(name, seed=seed)
