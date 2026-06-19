from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Return the repository root path for the hotel booking copilot project."""
    return Path(__file__).resolve().parent.parent


def get_data_dir(root: Optional[Path] = None) -> Path:
    """Return the project data directory path."""
    root_path = root or get_project_root()
    return root_path / "data"


def download_dataset(url: str, target_path: Path, overwrite: bool = False) -> Path:
    """Download the dataset from a public URL into the raw data directory.

    Args:
        url: Remote dataset URL.
        target_path: Local destination path.
        overwrite: Whether to overwrite an existing file.

    Returns:
        Path to the downloaded CSV file.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite:
        logger.info("Dataset already exists at %s", target_path)
        return target_path

    logger.info("Downloading dataset from %s", url)
    urllib.request.urlretrieve(url, target_path)
    logger.info("Saved dataset to %s", target_path)
    return target_path


def load_hotel_booking_data(csv_path: Path | str) -> pd.DataFrame:
    """Load hotel booking data from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def get_default_hotel_booking_path(root: Optional[Path] = None) -> Path:
    """Return the default CSV path for the hotel booking dataset."""
    return get_data_dir(root) / "hotel_bookings.csv"
