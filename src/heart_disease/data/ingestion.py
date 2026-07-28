from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame

    Parameters
    ----------
    path: Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    return pd.read_csv(path)

    