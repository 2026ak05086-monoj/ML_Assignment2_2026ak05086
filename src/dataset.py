"""Download and load the UCI dataset #697.

"Predict Students' Dropout and Academic Success" is distributed as a single
zipped, semicolon-separated table. We fetch it once, cache a tidy comma-
separated copy under ``data/raw/`` and expose small loaders for the rest of the
pipeline. Re-running is cheap because the cached CSV is reused.

Run as a module from the project root::

    python -m src.dataset
"""
from __future__ import annotations

import io
import zipfile
from urllib.request import Request, urlopen

import pandas as pd

from src import config


def _download_archive() -> bytes:
    """Fetch the raw ``.zip`` archive from the UCI repository."""
    request = Request(config.UCI_ARCHIVE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=90) as response:  # noqa: S310 (trusted host)
        return response.read()


def _read_table_from_archive(archive_bytes: bytes) -> pd.DataFrame:
    """Extract the single data table from the downloaded archive."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as bundle:
        csv_members = [n for n in bundle.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError("No .csv table found inside the UCI archive.")
        # The dataset ships one table; pick the largest csv just to be safe.
        largest = max(csv_members, key=lambda n: bundle.getinfo(n).file_size)
        payload = bundle.read(largest)

    # UCI publishes this file with ';' separators; fall back to ',' if needed.
    frame = pd.read_csv(io.BytesIO(payload), sep=";")
    if frame.shape[1] == 1:
        frame = pd.read_csv(io.BytesIO(payload), sep=",")
    return frame


def fetch_raw(force: bool = False) -> pd.DataFrame:
    """Return the raw dataset, downloading and caching it on first use."""
    config.ensure_directories()
    if config.RAW_CSV.exists() and not force:
        return pd.read_csv(config.RAW_CSV)

    frame = _read_table_from_archive(_download_archive())
    frame.columns = [col.strip() for col in frame.columns]
    frame.to_csv(config.RAW_CSV, index=False)
    return frame


def load_raw() -> pd.DataFrame:
    """Load the cached dataset, fetching it first if it is missing."""
    if not config.RAW_CSV.exists():
        return fetch_raw()
    return pd.read_csv(config.RAW_CSV)


def _summarise(frame: pd.DataFrame) -> None:
    """Print a short, human-readable snapshot of the loaded data."""
    n_rows, n_cols = frame.shape
    print(f"Rows: {n_rows} | Columns: {n_cols} ({n_cols - 1} features + target)")
    print(f"Cached at: {config.RAW_CSV}")
    print(f"Missing cells: {int(frame.isna().sum().sum())}")
    print(f"Duplicate rows: {int(frame.duplicated().sum())}")
    if config.TARGET_COLUMN in frame.columns:
        print("\nTarget distribution:")
        print(frame[config.TARGET_COLUMN].value_counts())
    else:
        print(f"\n[warn] Expected target column '{config.TARGET_COLUMN}' not found.")
        print("First columns:", list(frame.columns[:5]))


if __name__ == "__main__":
    data = fetch_raw(force=True)
    _summarise(data)
