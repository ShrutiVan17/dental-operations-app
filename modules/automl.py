# modules/automl.py — sklearn AutoML-lite (Py 3.12 friendly)
# - Trains a few lightweight candidates and picks the best
# - Saves model + feature order
# - Supports threshold for no-show scoring
# - Exports feature importances when model supports them

import os, json
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

NO_SHOW_BUNDLE = os.path.join(MODELS_DIR, "no_show_clf_sklearn.joblib")
NO_SHOW_META   = os.path.join(MODELS_DIR, "no_show_meta.json")
NO_SHOW_IMP    = os.path.join(MODELS_DIR, "no_show_importance.csv")

COLL_BUNDLE    = os.path.join(MODELS_DIR, "collections_reg_sklearn.joblib")
COLL_IMP       = os.path.join(MODELS_DIR, "collections_importance.csv")


# ----------------------------- utilities -----------------------------

def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns to numeric where possible (others stay as-is)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _clf_candidates(random_state=42):
    """Lightweight classifiers that work on 3.12 without extras."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }

def _reg_candidates(random_state=42):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    return {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
    }

def _maybe_feature_importance(estimator, feature_names: list[str]) -> pd.DataFrame | None:
    """Return importances/abs(coef) dataframe if the estimator supports it; else None."""
    if hasattr(estimator, "feature_importances_"):
        imp = np.asarray(estimator.feature_importances_)
        return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    if hasattr(estimator, "coef_"):
        coefs = np.asarray(estimator.coef_).ravel()
        return pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(coefs),
            "coef": coefs
        }).sort_values("importance", ascending=False)
    return None


# ----------------------- NO-SHOW (classification) -----------------------

def train_no_show(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df_features to include 'target_no_show' + predictor columns.
    Tries a few models, returns a leaderboard, and saves the best to disk
    along with feature order and basic metrics.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

    df_features = _safe_numeric(df_features)
    X = df_features.drop(columns=["target_no_show"])
    y = df_features["target_no_show"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = _clf_candidates()
    leaderboard = []

    best_auc = -1.0
    best_pipe: Pipeline | None = None
    best_name = None

    for name, base in candidates.items():
        steps = []
        # only scale for LR; tree-based models don't need it
        if name == "LogisticRegression":
            steps.append(("scaler", StandardScaler(with_mean=False)))
        steps.append(("clf", base))
        pipe = Pipeline(steps)

        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred05 = (proba >= 0.5).astype(int)

        auc = roc_auc_score(yte, proba)
        acc = accuracy_score(yte, pred05)
        f1  = f1_score(yte, pred05, zero_division=0)
        rec = recall_score(yte, pred05, zero_division=0)
        pre = precision_score(yte, pred05, zero_division=0)

        leaderboard.append({
            "model": name,
            "AUC": float(auc),
            "Accuracy": float(acc),
            "F1@0.5": float(f1),
            "Recall@0.5": float(rec),
            "Precision@0.5": float(pre),
            "backend": "sklearn",
        })

        if auc > best_auc:
            best_auc = auc
            best_pipe = pipe
            best_name = name

    # Save best pipeline with feature order
    assert best_pipe is not None, "No classifier was trained."
    joblib.dump({"model": best_pipe, "features": X.columns.tolist(), "name": best_name}, NO_SHOW_BUNDLE)

    # Save importances if available
    try:
        est = best_pipe.named_steps["clf"]
        imp_df = _maybe_feature_importance(est, X.columns.tolist())
        if imp_df is not None:
            imp_df.to_csv(NO_SHOW_IMP, index=False)
    except Exception:
        pass

    # Save quick meta metrics for display
    try:
        with open(NO_SHOW_META, "w", encoding="utf-8") as f:
            json.dump({"best_model": best_name, "best_auc": float(best_auc)}, f)
    except Exception:
        pass

    return pd.DataFrame(leaderboard).sort_values("AUC", ascending=False).reset_index(drop=True)


def score_no_show(rows: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Scores rows using the saved classifier.

    Returns a DataFrame with:
      - 'Score': predicted probability of no-show (0..1)
      - 'Label': 1 if Score >= threshold else 0
    """
    rows = _safe_numeric(rows)
    bundle = joblib.load(NO_SHOW_BUNDLE)
    pipe = bundle["model"]
    feats = bundle.get("features", list(rows.columns))
    X = rows.reindex(columns=feats, fill_value=0)

    proba = pipe.predict_proba(X)[:, 1]
    score = proba.astype(float)
    label = (score >= float(threshold)).astype(int)

    # IMPORTANT: don't return the original feature columns
    return pd.DataFrame({"Score": score, "Label": label})


# ----------------------- COLLECTIONS (regression) -----------------------

def train_collections(df_features: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Tries two regressors and saves the best with feature order.
    Returns a leaderboard with R2 and MAE.
    """
    from sklearn.model_selection import train_test_split

    df_features = _safe_numeric(df_features)
    X = df_features.drop(columns=[target_col])
    y = pd.to_numeric(df_features[target_col], errors="coerce").fillna(0.0)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.metrics import r2_score, mean_absolute_error

    models = _reg_candidates()
    leaderboard = []

    best_r2 = -1e9
    best_model = None
    best_name = None

    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        r2 = r2_score(yte, pred)
        mae = mean_absolute_error(yte, pred)
        leaderboard.append({"model": name, "R2": float(r2), "MAE": float(mae), "backend": "sklearn"})
        if r2 > best_r2:
            best_r2 = r2
            best_model = mdl
            best_name = name

    assert best_model is not None, "No regressor was trained."
    joblib.dump({"model": best_model, "features": X.columns.tolist(), "name": best_name, "target": target_col}, COLL_BUNDLE)

    # Save importances if available
    try:
        imp_df = _maybe_feature_importance(best_model, X.columns.tolist())
        if imp_df is not None:
            imp_df.to_csv(COLL_IMP, index=False)
    except Exception:
        pass

    return pd.DataFrame(leaderboard).sort_values("R2", ascending=False).reset_index(drop=True)


def score_collections(rows: pd.DataFrame) -> pd.DataFrame:
    rows = _safe_numeric(rows)
    bundle = joblib.load(COLL_BUNDLE)
    model = bundle["model"]
    feats = bundle.get("features", list(rows.columns))
    X = rows.reindex(columns=feats, fill_value=0)

    val = model.predict(X).astype(float)

    # Again: only return the prediction column
    return pd.DataFrame({"Score": val})
