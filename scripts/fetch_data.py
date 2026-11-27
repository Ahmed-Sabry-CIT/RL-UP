"""Download and validate the three benchmark datasets into ``data/raw``.

Usage
-----
    python scripts/fetch_data.py                 # all three datasets
    python scripts/fetch_data.py --dataset fraud # only one
    python scripts/fetch_data.py --check         # verify what is already there

Sources (public, no credentials):
* healthcare -- UCI ML Repository 296, Diabetes 130-US hospitals (zip)
* fraud      -- ULB Credit Card Fraud Detection, OpenML 1597 (CSV gzip)
* network    -- NSL-KDD KDDTrain+ / KDDTest+ (plain text)

Every download is validated (row/column sanity check) before being accepted, so
a truncated file or an HTML error page is reported instead of silently loaded.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

SOURCES = {
    "healthcare": [
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/"
            "dataset_diabetes.zip",
            "dataset_diabetes.zip",
        )
    ],
    "fraud": [
        ("https://www.openml.org/data/get_csv/1673544/phpKo8OWT.csv.gz", "creditcard.csv"),
        ("https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv", "creditcard.csv"),
    ],
    "network": [
        ("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt", "KDDTrain+.txt"),
        ("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt", "KDDTest+.txt"),
    ],
}

MIN_BYTES = {"healthcare": 1_000_000, "fraud": 5_000_000, "network": 1_000_000}


def _download(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=180) as response, open(destination, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        if destination.exists():
            destination.unlink()
        return False
    size = destination.stat().st_size
    print(f"  saved {destination.name} ({size / 1e6:.1f} MB)")
    return True


def _validate(dataset: str) -> tuple[bool, str]:
    if dataset == "healthcare":
        csv_path = RAW / "diabetic_data.csv"
        if not csv_path.exists():
            for archive in RAW.glob("*.zip"):
                try:
                    with zipfile.ZipFile(archive) as zf:
                        member = next(
                            (n for n in zf.namelist() if n.endswith("diabetic_data.csv")), None
                        )
                        if member:
                            with zf.open(member) as src, open(csv_path, "wb") as dst:
                                dst.write(src.read())
                            break
                except zipfile.BadZipFile:
                    return False, f"{archive.name} is not a valid zip"
        if not csv_path.exists():
            return False, "diabetic_data.csv not found"
        with open(csv_path, encoding="utf-8", errors="ignore") as handle:
            header = handle.readline().strip().split(",")
        required = {"encounter_id", "patient_nbr", "readmitted"}
        if not required <= set(header):
            return False, f"unexpected healthcare header ({len(header)} columns)"
        return True, f"diabetic_data.csv with {len(header)} columns"

    if dataset == "fraud":
        for name in ("creditcard.csv", "creditcard.csv.gz"):
            path = RAW / name
            if not path.exists():
                continue
            compression = None
            if path.suffix == ".gz":
                with open(path, "rb") as probe:
                    compression = "gzip" if probe.read(2) == b"\x1f\x8b" else None
            opener = gzip.open if compression == "gzip" else open
            with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
                header = handle.readline().strip().replace('"', "").split(",")
                second = handle.readline()
            if "Class" not in header or len(header) < 30:
                return False, f"{name}: unexpected header {header[:5]}..."
            if not second:
                return False, f"{name}: no data rows"
            return True, f"{name} with {len(header)} columns"
        return False, "creditcard.csv[.gz] not found"

    if dataset == "network":
        found = [p for p in (RAW / "KDDTrain+.txt", RAW / "KDDTest+.txt") if p.exists()]
        if not found:
            return False, "KDDTrain+.txt / KDDTest+.txt not found"
        total = 0
        for path in found:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                first = handle.readline().strip().split(",")
                total += 1 + sum(1 for _ in handle)
            if len(first) != 43:
                return False, f"{path.name}: {len(first)} columns (expected 43)"
        return True, f"{', '.join(p.name for p in found)} ({total} rows)"

    return False, "unknown dataset"


def fetch(dataset: str) -> bool:
    print(f"[{dataset}]")
    ok, detail = _validate(dataset)
    if ok:
        print(f"  already present: {detail}")
        return True

    for url, filename in SOURCES[dataset]:
        destination = RAW / filename
        if destination.exists() and destination.stat().st_size >= MIN_BYTES[dataset]:
            continue
        if not _download(url, destination):
            continue
        ok, detail = _validate(dataset)
        if ok:
            print(f"  validated: {detail}")
            return True
        print(f"  validation failed: {detail}")
        if destination.exists() and destination.stat().st_size < MIN_BYTES[dataset]:
            destination.unlink()

    ok, detail = _validate(dataset)
    print(f"  NOT AVAILABLE: {detail}")
    print(f"  -> download {dataset} manually into {RAW}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", help="all | healthcare,fraud,network")
    parser.add_argument("--check", action="store_true", help="only validate local files")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    names = list(SOURCES) if args.dataset == "all" else [
        item.strip() for item in args.dataset.split(",") if item.strip()
    ]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}")

    status = {}
    for name in names:
        if args.check:
            ok, detail = _validate(name)
            print(f"[{name}] {'OK' if ok else 'MISSING'} -- {detail}")
        else:
            ok = fetch(name)
        status[name] = ok

    print()
    for name, ok in status.items():
        print(f"{'READY ' if ok else 'MISSING'} {name}")
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
