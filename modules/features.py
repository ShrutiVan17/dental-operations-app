# modules/features.py

from __future__ import annotations
import pandas as pd
from typing import Dict, Any

# Source columns (adjust if your CSV uses different names)
DATE_COL        = "DateKpi"
NO_SHOW_COL     = "VisitsNoShow"
HYG_COL         = "VisitsHygieneCompleted"
REST_COL        = "VisitsRestorativeCompleted"
COLLECTIONS_COL = "Collections"
PROFIT_COL      = None  # set your profit column name if you have one

def _coerce_num(s: pd.Series) -> pd.Series:
    if s.dtype == "O":
        s = (
            s.astype(str)
             .str.replace(r"[,$]", "", regex=True)
             .str.strip()
        )
    return pd.to_numeric(s, errors="coerce")

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Date features
    if DATE_COL in out.columns:
        d = pd.to_datetime(out[DATE_COL], errors="coerce")
    else:
        d = pd.to_datetime("today")

    out["date"] = d.dt.floor("D")
    out["dayofweek"] = out["date"].dt.dayofweek.fillna(0).astype(int)
    out["month"] = out["date"].dt.month.fillna(1).astype(int)
    out["year"] = out["date"].dt.year.fillna(out["date"].dt.year.mode().iloc[0] if len(out) else pd.Timestamp.today().year).astype(int)

    # Binary flags
    if HYG_COL in out.columns:
        out["is_hygiene"] = (_coerce_num(out[HYG_COL]).fillna(0) > 0).astype(int)
    else:
        out["is_hygiene"] = 0

    if REST_COL in out.columns:
        out["is_restorative"] = (_coerce_num(out[REST_COL]).fillna(0) > 0).astype(int)
    else:
        out["is_restorative"] = 0

    # No-show target
    if NO_SHOW_COL in out.columns:
        out["target_no_show"] = (_coerce_num(out[NO_SHOW_COL]).fillna(0) > 0).astype(int)
    else:
        out["target_no_show"] = 0

    # Optional numerics
    if COLLECTIONS_COL and COLLECTIONS_COL in out.columns:
        out["Collections"] = _coerce_num(out[COLLECTIONS_COL])
    if PROFIT_COL and PROFIT_COL in out.columns:
        out["Profit"] = _coerce_num(out[PROFIT_COL])

    return out

def appt_row_to_features(row: pd.Series) -> Dict[str, Any]:
    """
    Map one appointment row to model features:
      dayofweek, month, is_hygiene, is_restorative
    Uses flags if present; otherwise infers from free-text 'reason'.
    """
    d = pd.to_datetime(row.get("date"), errors="coerce")

    is_hyg = int(row.get("is_hygiene", 0))
    is_res = int(row.get("is_restorative", 0))

    if (is_hyg == 0 or is_res == 0) and "reason" in row.index:
        reason = str(row.get("reason", "")).lower()
        if is_hyg == 0 and any(k in reason for k in ["hyg", "clean", "prophy", "recall"]):
            is_hyg = 1
        if is_res == 0 and any(k in reason for k in ["rest", "fill", "crown", "endo", "root", "extraction", "implant"]):
            is_res = 1

    return {
        "dayofweek": int(d.dayofweek) if pd.notna(d) else 0,
        "month":     int(d.month)     if pd.notna(d) else 1,
        "is_hygiene": is_hyg,
        "is_restorative": is_res,
    }

__all__ = ["add_basic_features", "appt_row_to_features"]
