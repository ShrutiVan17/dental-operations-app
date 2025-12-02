
import pandas as pd
import numpy as np
from typing import List, Optional

def combine_columns(df: pd.DataFrame, cols: List[str], method: str = "sum", weights: Optional[list] = None, new_name: str = "CollectionsCombined"):
    if not cols:
        raise ValueError("No columns selected.")
    d = df.copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if method == "sum":
        d[new_name] = d[cols].sum(axis=1, skipna=True)
    elif method == "avg":
        d[new_name] = d[cols].mean(axis=1, skipna=True)
    elif method == "weighted":
        if not weights or len(weights) != len(cols):
            raise ValueError("Weights must be the same length as columns for weighted method.")
        w = np.array(weights, dtype=float)
        M = d[cols].to_numpy(dtype=float)
        d[new_name] = np.nansum(M * w, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")
    return d
