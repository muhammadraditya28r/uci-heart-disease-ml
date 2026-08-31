from collections.abc import Callable

from pathlib import Path
import pandas as pd



def load_file(path: str | Path,
              read_func: Callable | None = None,
              ) -> pd.DataFrame:
    """
    Load a file into a pandas DataFrame

    Parameters
    ----------
    path: Path to the file.
    read_func: pandas read file method

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    if read_func == None:        
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix == ".parquet":
            return pd.read_parquet(path)
        
        raise ValueError(f"Unsupported file type {suffix}, try using read_func")

    else: 
        return read_func(path)

