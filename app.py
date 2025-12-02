# app.py — Dental Scheduler + Prediction (sklearn, Python 3.12 friendly)

import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date

# ---- our modules (no PyCaret) ----
from modules.features import add_basic_features
from modules.automl import (
    train_no_show, score_no_show,
    train_collections, score_collections,
)
# ----- HARDENED FEATURES IMPORT (put this right after your std imports) -----
import importlib.util, pathlib

# Try to import only add_basic_features from modules.features.
# If it fails, define a local fallback.
try:
    from modules.features import add_basic_features
except Exception:
    import pandas as pd

    # Minimal local fallback (mirrors your modules/features.py)
    DATE_COL, NO_SHOW_COL = "DateKpi", "VisitsNoShow"
    HYG_COL, REST_COL = "VisitsHygieneCompleted", "VisitsRestorativeCompleted"
    COLLECTIONS_COL, PROFIT_COL = "Collections", None

    def _coerce_num(s: pd.Series) -> pd.Series:
        if s.dtype == "O":
            s = s.astype(str).str.replace(r"[,$]", "", regex=True).str.strip()
        return pd.to_numeric(s, errors="coerce")

    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if DATE_COL in out.columns:
            d = pd.to_datetime(out[DATE_COL], errors="coerce")
        else:
            d = pd.to_datetime("today")
        out["date"] = d.dt.floor("D")
        out["dayofweek"] = out["date"].dt.dayofweek.fillna(0).astype(int)
        out["month"] = out["date"].dt.month.fillna(1).astype(int)
        out["year"] = out["date"].dt.year.fillna(pd.Timestamp.today().year).astype(int)

        out["is_hygiene"] = (_coerce_num(out[HYG_COL]).fillna(0) > 0).astype(int) if HYG_COL in out.columns else 0
        out["is_restorative"] = (_coerce_num(out[REST_COL]).fillna(0) > 0).astype(int) if REST_COL in out.columns else 0
        out["target_no_show"] = (_coerce_num(out[NO_SHOW_COL]).fillna(0) > 0).astype(int) if NO_SHOW_COL in out.columns else 0
        if COLLECTIONS_COL in out.columns:
            out["Collections"] = _coerce_num(out[COLLECTIONS_COL])
        if PROFIT_COL and PROFIT_COL in out.columns:
            out["Profit"] = _coerce_num(out[PROFIT_COL])
        return out
# ----- END HARDENED FEATURES IMPORT -----

# ---- Optional helpers (safe if missing) ----
try:
    from modules.plots import timeseries_line, bar_by, year_month_breakdown
except Exception:
    timeseries_line = bar_by = year_month_breakdown = None

try:
    from modules.calc import combine_columns
except Exception:
    combine_columns = None

try:
    from modules.scraper import scrape_public_pages
except Exception:
    scrape_public_pages = None

try:
    from modules.llm_helper import draft_strategies
except Exception:
    draft_strategies = None

try:
    from modules.chatbot import build_context_cards, chat_answer
except Exception:
    build_context_cards = chat_answer = None


# ---------------------------------------------------------------------
# Local shim so we DO NOT depend on importing appt_row_to_features
# from modules.features (avoids your earlier import error).
# ---------------------------------------------------------------------
def appt_row_to_features(row: pd.Series) -> dict:
    """
    Map one appointment row to model features:
      dayofweek, month, is_hygiene, is_restorative
    Uses flags when present; else infers from free-text 'reason'.
    """
    d = pd.to_datetime(row.get("date"), errors="coerce")

    # flags if present
    is_hyg = int(row.get("is_hygiene", 0))
    is_res = int(row.get("is_restorative", 0))

    # fallback: infer from text
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


# --------------- App Config --------------------
st.set_page_config(
    page_title="Dental Scheduler + Prediction",
    layout="wide",
)
DATA_PATH = "data/datasetcleaned.csv"

st.title("🦷 Dental Operations")
# -------------------- Simple Login State --------------------
USERS = {
    "owner": {"password": "owner123", "role": "Owner"},
    "reception": {"password": "rec123", "role": "Reception"},
    "manager": {"password": "manager123", "role": "Manager"},
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None


# -------------------- Sidebar / Login + Data + Page --------------------
with st.sidebar:
    # -------- Login section at top --------
    st.header("Login")

    if not st.session_state.logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("🔐 Login"):
            user = USERS.get(username.lower())
            if user and password == user["password"]:
                st.session_state.logged_in = True
                st.session_state.role = user["role"]
                st.session_state.username = username
                st.success(f"Logged in as {st.session_state.role}")
                st.rerun()
            else:
                st.error("Invalid username or password")

        # stop the app here until they log in
        st.stop()

    else:
        st.write(
            f"Logged in as **{st.session_state.username}** "
            f"({st.session_state.role})"
        )
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.username = None
            st.rerun()

    st.markdown("---")

    # -------- Data upload --------
    st.header("Data")
    up = st.file_uploader("Upload CSV (or use bundled data/datasetcleaned.csv)", type=["csv"])
    if up:
        df_raw = pd.read_csv(up)
    else:
        if not os.path.exists(DATA_PATH):
            st.error("Missing data/datasetcleaned.csv. Upload a CSV on the left.")
            st.stop()
        df_raw = pd.read_csv(DATA_PATH)

    st.caption(f"Rows: {len(df_raw):,} | Columns: {len(df_raw.columns)}")
    st.markdown("---")

    # simple role router (Owner sees Prediction by default, Receptionist sees Scheduler)
    role = st.session_state.role or st.selectbox("Role", ["Owner", "Receptionist"], index=0)
    default_page = "🤖 Prediction" if role == "Owner" else "🗓️ Scheduler"

    mode = st.radio(
        "Page",
        ["🤖 Prediction", "🗓️ Scheduler", "📈 Collections & Profit",
         "🧮 Collections Calculator", "🧠 Strategies (LLM)", "❓ Data Q&A"],
        index=["🤖 Prediction", "🗓️ Scheduler", "📈 Collections & Profit",
               "🧮 Collections Calculator", "🧠 Strategies (LLM)", "❓ Data Q&A"].index(default_page)
    )

# Build engineered features ONCE
df = df_raw.copy()
df_feat = add_basic_features(df)
# Make sure we have a proper date column & a consistent baseline
if "date" in df_feat.columns:
    df_feat["date"] = pd.to_datetime(df_feat["date"], errors="coerce")
    BASELINE_DATE = df_feat["date"].max().normalize()
else:
    BASELINE_DATE = pd.Timestamp.today().normalize()


# ======================== 🤖 Prediction ========================
if mode == "🤖 Prediction":
    st.subheader("Prediction: no-show risk & revenue impact")

    st.caption(
        "Use this page in two steps:"
        " 1) Train the models once on your historical data; "
        " 2) Simulate an appointment and see an easy-to-read risk & revenue story."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "1️⃣ Train no-show model (setup)",
            "2️⃣ Train collections model (setup)",
            "3️⃣ Simulate an appointment (owner view)",
        ]
    )

    # ---- Tab 1: Train No-Show (classification) ----
    with tab1:
        st.markdown(
            "**What this does (in plain English):**\n\n"
            "- Looks at your past appointments.\n"
            "- Learns when patients tend to **not show up**.\n"
            "- Later, we use it to flag risky appointments as **Low / Medium / High** no-show risk."
        )

        needed = ["dayofweek", "month", "is_hygiene", "is_restorative", "target_no_show"]
        miss = [c for c in needed if c not in df_feat.columns]
        if miss:
            st.warning(f"Missing columns for no-show training: {miss}")
        else:
            if st.button("🚀 Train / retrain no-show model", key="btn_train_no_show"):
                with st.spinner("Training on your historical appointments..."):
                    # train_no_show expects a dataframe with features + target_no_show
                    lb = train_no_show(df_feat[needed])

                st.success("No-show model saved: models/no_show_clf_sklearn.joblib")

                with st.expander("See technical training details (optional)"):
                    st.dataframe(lb, use_container_width=True)
                    st.caption(
                        "This table is mainly for your data team. As an owner, "
                        "you only need to know the model has been trained."
                    )

                if "target_no_show" in df_feat.columns:
                    st.caption(
                        f"Historical no-show rate in your data: "
                        f"**{df_feat['target_no_show'].mean():.1%}**"
                    )

    # ---- Tab 2: Train Collections (regression) ----
    with tab2:
        st.markdown(
            "**What this does (in plain English):**\n\n"
            "- Learns how much money a typical appointment brings in, "
            "depending on the day and type of visit.\n"
            "- Later, we use it to estimate **expected collections** for a specific appointment."
        )

        needed_reg = ["dayofweek", "month", "is_hygiene", "is_restorative", "Collections"]
        miss2 = [c for c in needed_reg if c not in df_feat.columns]
        if miss2:
            st.warning(f"Missing columns for collections training: {miss2}")
        else:
            if st.button("💰 Train / retrain collections model", key="btn_train_collections"):
                with st.spinner("Learning from your past collections..."):
                    # train_collections now takes the dataframe AND target column name
                    lb2 = train_collections(
                        df_feat[needed_reg],
                        target_col="Collections",
                    )

                st.success("Collections model saved: models/collections_reg_sklearn.joblib")

                with st.expander("See technical training details (optional)"):
                    st.dataframe(lb2, use_container_width=True)

                if "Collections" in df_feat.columns:
                    st.caption(
                        f"Average collections per row in your data: "
                        f"**${df_feat['Collections'].mean():,.0f}**"
                    )

   

    # ---- Tab 3: Score / What-if (owner view, MODEL + DATA) ----
    with tab3:
        st.markdown(
            "Fill in the appointment details below. We’ll show:\n"
            "- **Model-estimated no-show risk** using your trained ML model.\n"
            "- **Model-estimated expected collections** using your trained regression model.\n"
            "- A comparison with **similar visits in your historical data** so it is easy to trust."
        )

        # ------------- INPUTS -------------
        col_left, col_right = st.columns(2)

        with col_left:
            appt_date = st.date_input("📅 Appointment date", value=date.today())
            appt_type = st.selectbox(
                "Type of visit",
                ["Hygiene (cleaning)", "Restorative (filling / crown)", "Mixed", "Other"],
            )
            reason = st.text_input(
                "Optional notes (reason for visit)",
                value="Recall cleaning",
                help="Purely for description; not used directly in the model.",
            )

        with col_right:
            st.markdown("**How we’ll treat this visit:**")

            if appt_type.startswith("Hygiene"):
                is_hyg, is_res = True, False
            elif appt_type.startswith("Restorative"):
                is_hyg, is_res = False, True
            elif appt_type == "Mixed":
                is_hyg, is_res = True, True
            else:
                is_hyg, is_res = False, False

            st.write(f"- Hygiene work: **{'Yes' if is_hyg else 'No'}**")
            st.write(f"- Restorative work: **{'Yes' if is_res else 'No'}**")

        # Build feature row (for model + data filtering)
        row = {
            "date": pd.to_datetime(appt_date),
            "is_hygiene": int(is_hyg),
            "is_restorative": int(is_res),
            "reason": reason,
        }
        feat_row = appt_row_to_features(pd.Series(row))

        # Helper names for story
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = dow_names[feat_row["dayofweek"]] if 0 <= feat_row["dayofweek"] <= 6 else "Unknown day"
        month_name = row["date"].strftime("%B")

        st.markdown(
            f"**We see this as:** a visit on **{day_name} in {month_name}** "
            f"({'hygiene' if feat_row['is_hygiene'] else 'no hygiene'}, "
            f"{'restorative' if feat_row['is_restorative'] else 'no restorative'})."
        )

        # ------------- PREDICT BUTTON -------------
        if st.button("🔮 Predict risk & revenue", key="predict_hybrid"):
            # ============= 1. MODEL-BASED PREDICTION (ML) =============
            X_one = pd.DataFrame([feat_row])

            p_no_show_model = None
            pred_coll_model = None

            # Helper: convert different return types -> single float
            import numpy as np

            def to_float(val):
                """Convert DataFrame/Series/array/dict → single float or None."""
                if val is None:
                    return None
                if isinstance(val, (int, float)):
                    return float(val)

                if isinstance(val, pd.DataFrame):
                    num_cols = val.select_dtypes("number").columns
                    if len(num_cols):
                        return float(val[num_cols[0]].iloc[0])
                    return None

                if isinstance(val, pd.Series):
                    if len(val):
                        return float(val.iloc[0])
                    return None

                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.array(val).ravel()
                    if arr.size:
                        return float(arr[0])
                    return None

                if isinstance(val, dict):
                    for key in ("p_no_show", "prob", "proba", "Score", "prediction", "pred", "value"):
                        if key in val:
                            try:
                                return float(val[key])
                            except Exception:
                                continue
                    return None

                return None

            # --- No-show prediction via ML model ---
            try:
                raw_ns = score_no_show(X_one)  # use your trained classifier
                p_no_show_model = to_float(raw_ns)
            except Exception as e:
                st.error(f"No-show model prediction error: {e}")

            # --- Collections prediction via ML model ---
            try:
                raw_coll = score_collections(X_one)  # use your trained regressor
                pred_coll_model = to_float(raw_coll)
            except Exception as e:
                st.error(f"Collections model prediction error: {e}")

            # Normalise probability to [0,1]
            if p_no_show_model is not None:
                if p_no_show_model > 1 and p_no_show_model <= 100:
                    p_no_show_model = p_no_show_model / 100.0
                p_no_show_model = max(0.0, min(1.0, p_no_show_model))

            # ------------- 2. DATA-ONLY HISTORICAL VIEW -------------
            similar = df_feat.copy()
            if "dayofweek" in similar.columns:
                similar = similar[similar["dayofweek"] == feat_row["dayofweek"]]
            if "is_hygiene" in similar.columns:
                similar = similar[similar["is_hygiene"] == feat_row["is_hygiene"]]
            if "is_restorative" in similar.columns:
                similar = similar[similar["is_restorative"] == feat_row["is_restorative"]]

            # If that’s too narrow, fall back to same day-of-week only
            if similar.empty and "dayofweek" in df_feat.columns:
                similar = df_feat[df_feat["dayofweek"] == feat_row["dayofweek"]]

            # If still empty, fall back to the whole dataset
            if similar.empty:
                similar = df_feat.copy()

            n_sim = len(similar)

            hist_no_show = None
            hist_coll = None

            if "target_no_show" in similar.columns:
                no_show_series = pd.to_numeric(similar["target_no_show"], errors="coerce")
                if no_show_series.notna().any():
                    hist_no_show = float(no_show_series.mean())

            if "Collections" in similar.columns:
                coll_series = pd.to_numeric(similar["Collections"], errors="coerce")
                if coll_series.notna().any():
                    hist_coll = float(coll_series.mean())

            # ------------- 3. OWNER-FRIENDLY OUTPUT -------------
            # Risk bands (based on model where available, otherwise historical)
            p_for_band = p_no_show_model if p_no_show_model is not None else hist_no_show

            risk_label = "Unknown"
            risk_expl = ""
            if p_for_band is not None:
                p_for_band = max(0.0, min(1.0, p_for_band))
                if p_for_band < 0.15:
                    risk_label = "Low"
                    risk_expl = "Most similar visits usually show up."
                elif p_for_band < 0.35:
                    risk_label = "Medium"
                    risk_expl = "Some risk of no-show. Consider a reminder."
                else:
                    risk_label = "High"
                    risk_expl = "High risk of no-show. Consider double-confirmation or overbooking."

            c1, c2 = st.columns(2)

            # --- LEFT: model-based view ---
            with c1:
                st.markdown("#### Model estimate (trained on your data)")
                if p_no_show_model is not None:
                    st.metric(
                        "No-show risk (model)",
                        f"{p_no_show_model:.1%}",
                        help="Predicted by the ML model trained on your historical appointments.",
                    )
                else:
                    st.write("No-show model not available or failed.")

                if pred_coll_model is not None:
                    st.metric(
                        "Expected collections (model)",
                        f"${pred_coll_model:,.0f}",
                        help="Predicted by the ML model trained on your historical collections.",
                    )
                else:
                    st.write("Collections model not available or failed.")

                if p_for_band is not None:
                    st.markdown(f"**Risk level:** {risk_label}  \n{risk_expl}")

            # --- RIGHT: historical data view ---
            with c2:
                st.markdown("#### What your history says (similar visits)")
                if hist_no_show is not None:
                    st.metric(
                        "No-show rate (history)",
                        f"{hist_no_show:.1%}",
                        help="Average no-show rate for similar visits in your past data.",
                    )
                else:
                    st.write("No-show information not available in your data.")

                if hist_coll is not None:
                    st.metric(
                        "Average collections (history)",
                        f"${hist_coll:,.0f}",
                        help="Average collections for similar visits in your past data.",
                    )
                else:
                    st.write("Collections information not available in your data.")

                st.caption(f"Similar visits found in history: **{n_sim:,}**")

            # ------------- 4. STORY FOR OWNER -------------
            st.markdown("### Why do we show these numbers?")

            bullets = []
            bullets.append(
                f"- We found **{n_sim:,} past visits** like this "
                f"({day_name}, same type of work)."
            )
            if hist_no_show is not None:
                bullets.append(
                    f"- Historically, **{hist_no_show:.1%}** of those patients did **not** show up."
                )
            if hist_coll is not None:
                bullets.append(
                    f"- Those visits brought in about **${hist_coll:,.0f}** per appointment on average."
                )
            if p_no_show_model is not None:
                bullets.append(
                    f"- The **model**, trained on all your data, thinks this new appointment has "
                    f"about **{p_no_show_model:.1%}** chance of no-show."
                )
            if pred_coll_model is not None:
                bullets.append(
                    f"- If the patient comes, the model expects about **${pred_coll_model:,.0f}** in collections."
                )

            st.markdown("\n".join(bullets))

            st.markdown(
                "> **How to use this as an owner:**\n"
                "> - High no-show risk + low expected collections → consider overbooking or moving them.\n"
                "> - High no-show risk + high expected collections → put extra effort into reminders / confirmations.\n"
                "> - Low no-show risk → these are your most reliable slots."
            )


# ======================== 🗓️ Scheduler ========================
elif mode == "🗓️ Scheduler":
    from modules.llm_copy import draft_email_copy
    from modules.mailer import send_email

    st.subheader("Daily Schedule (front-desk)")

    APPT_PATH = "data/appointments.csv"
    os.makedirs("data", exist_ok=True)

    # ---------- helpers ----------
    def load_appts() -> pd.DataFrame:
        if os.path.exists(APPT_PATH):
            df = pd.read_csv(APPT_PATH)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
        # start empty the first time
        return pd.DataFrame(columns=[
            "id", "date", "time", "type", "patient", "email", "reason",
            "is_hygiene", "is_restorative", "status", "notes"
        ])

    def save_appts(df: pd.DataFrame):
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(APPT_PATH, index=False)

    def next_id(df: pd.DataFrame) -> int:
        return (int(df["id"].max()) + 1) if "id" in df.columns and len(df) else 1

    appts = load_appts()

    # ---------- tabs ----------
    t1, t2, t3 = st.tabs(["📅 Today", "➕ New / Reschedule", "🗑️ Cancel"])

    # ---------- 📅 Today ----------
    with t1:
        sel_date = st.date_input("Date", pd.Timestamp.today().date())
        day_df = appts.loc[pd.to_datetime(appts["date"]).dt.date == pd.to_datetime(sel_date).date()].copy()

        # Predict no-show risk if model exists
        if len(day_df):
            try:
                feats_rows = pd.DataFrame([appt_row_to_features(r) for _, r in day_df.iterrows()])
                scored = score_no_show(feats_rows, threshold=0.70)
                day_df = day_df.reset_index(drop=True).copy()
                day_df["no_show_risk"] = (scored["Score"] * 100).round(1)
            except Exception:
                day_df["no_show_risk"] = None

        st.dataframe(
            day_df.sort_values(["time", "type"]),
            use_container_width=True,
            column_config={
                "no_show_risk": st.column_config.NumberColumn("No-Show Risk (%)", format="%.1f"),
            }
        )

    # ---------- ➕ New / Reschedule ----------
    with t2:
        c1, c2 = st.columns(2)

        # New appointment
        with c1:
            st.markdown("**Create appointment**")
            new_date = st.date_input("Date", pd.Timestamp.today().date(), key="new_date")
            new_time = st.time_input("Time", pd.Timestamp.now().ceil("H").time(), key="new_time")
            appt_type = st.selectbox("Type", ["Hygiene", "Restorative"])
            patient = st.text_input("Patient full name")
            email = st.text_input("Patient email")
            notes = st.text_area("Notes (optional)", height=80)

            if st.button("Create appointment"):
                if not patient or not email:
                    st.error("Patient name and email are required.")
                else:
                    row = {
                        "id": next_id(appts),
                        "date": pd.to_datetime(new_date),
                        "time": new_time.strftime("%H:%M"),
                        "type": appt_type,
                        "patient": patient,
                        "email": email,
                        "reason": "Cleaning/Prophy" if appt_type == "Hygiene" else "Restorative care",
                        "is_hygiene": 1 if appt_type == "Hygiene" else 0,
                        "is_restorative": 1 if appt_type == "Restorative" else 0,
                        "status": "scheduled",
                        "notes": notes,
                    }
                    appts = pd.concat([appts, pd.DataFrame([row])], ignore_index=True)
                    save_appts(appts)

                    # Draft email with Ollama, then send
                    context = {"type": appt_type, "patient": patient,
                               "date": str(new_date), "time": row["time"], "status": "scheduled"}
                    subject, body = draft_email_copy(context)
                    ok, msg = send_email(email, subject, body)
                    st.success(f"Created appointment. Email: {'OK' if ok else msg}")

        # Reschedule
        with c2:
            st.markdown("**Reschedule appointment**")
            if len(appts) == 0:
                st.info("No appointments yet.")
            else:
                appts["label"] = appts.apply(
                    lambda r: f"#{int(r['id'])} • {r['patient']} • {pd.to_datetime(r['date']).date()} "
                              f"{r['time']} • {r['type']} • {r['status']}",
                    axis=1,
                )
                pick = st.selectbox("Select", options=appts["label"].tolist())
                row = appts.loc[appts["label"] == pick].iloc[0]

                rs_date = st.date_input("New date", pd.to_datetime(row["date"]).date(), key="rs_date")
                rs_time = st.time_input("New time", pd.to_datetime(row["time"]).time(), key="rs_time")

                if st.button("Reschedule"):
                    idx = appts.index[appts["id"] == row["id"]][0]
                    appts.at[idx, "date"] = pd.to_datetime(rs_date)
                    appts.at[idx, "time"] = rs_time.strftime("%H:%M")
                    appts.at[idx, "status"] = "rescheduled"
                    save_appts(appts)

                    context = {"type": row["type"], "patient": row["patient"],
                               "date": str(rs_date), "time": rs_time.strftime("%H:%M"), "status": "rescheduled"}
                    subject, body = draft_email_copy(context)
                    ok, msg = send_email(row["email"], subject, body)
                    st.success(f"Rescheduled. Email: {'OK' if ok else msg}")

    # ---------- 🗑️ Cancel ----------
    with t3:
        if len(appts) == 0:
            st.info("No appointments to cancel.")
        else:
            appts["label"] = appts.apply(
                lambda r: f"#{int(r['id'])} • {r['patient']} • {pd.to_datetime(r['date']).date()} "
                          f"{r['time']} • {r['type']} • {r['status']}",
                axis=1,
            )
            pick = st.selectbox("Select appointment to cancel", options=appts["label"].tolist(), key="cancel_pick")
            row = appts.loc[appts["label"] == pick].iloc[0]

            if st.button("Cancel appointment", type="primary"):
                idx = appts.index[appts["id"] == row["id"]][0]
                appts.at[idx, "status"] = "cancelled"
                save_appts(appts)

                context = {"type": row["type"], "patient": row["patient"],
                           "date": str(pd.to_datetime(row["date"]).date()),
                           "time": row["time"], "status": "cancelled"}
                subject, body = draft_email_copy(context)
                ok, msg = send_email(row["email"], subject, body)
                st.success(f"Cancelled. Email: {'OK' if ok else msg}")

# ======================== 📈 Collections & Profit ========================
elif mode == "📈 Collections & Profit":
    st.subheader("Owner Revenue Dashboard (simple view)")

    st.markdown(
        """
        This page tells a **simple revenue story** for the owner:

        1. Pick a **time window** (last 30 / 90 / 365 days or all data).  
        2. See **total collections** in that window and how it compares to the previous window.  
        3. Explore **trends over time** and **breakdowns** by weekday, month, and year.  

        All calculations are based on the data in your uploaded file.
        """
    )

    # --- basic checks ---
    if "date" not in df_feat.columns:
        st.error("No 'date' column found after feature engineering. Cannot build time-based dashboard.")
    elif not (timeseries_line and year_month_breakdown and bar_by):
        # fallback if plotting helpers missing
        st.info(
            "Optional plotting helpers not found (modules/plots.py). "
            "Showing a simple backup line chart instead."
        )
        df_tmp = df_feat.copy()
        df_tmp["date"] = pd.to_datetime(df_tmp["date"], errors="coerce")
        df_tmp = df_tmp.dropna(subset=["date"])
        df_tmp = df_tmp.sort_values("date")

        if "Collections" in df_tmp.columns:
            by_m = (
                df_tmp.groupby(df_tmp["date"].dt.to_period("M"))["Collections"]
                .sum()
                .to_timestamp()
            )
            st.line_chart(by_m, use_container_width=True)
        else:
            st.write(df_tmp.head())
    else:
        # --- clean + sort dates ---
        df_dash = df_feat.copy()
        df_dash["date"] = pd.to_datetime(df_dash["date"], errors="coerce")
        df_dash = df_dash.dropna(subset=["date"])
        df_dash = df_dash.sort_values("date")

        # optional: filter by practice so numbers look realistic per office
        practice_col = None
        for col in ["pratice_id", "practiceid", "PracticeId", "Practice"]:
            if col in df_dash.columns:
                practice_col = col
                break

        if practice_col is not None:
            practices = (
                df_dash[practice_col]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
            choice = st.selectbox(
                "Show data for which practice?",
                ["All practices"] + practices,
                index=0,
                help="Choose a single practice to see its own revenue numbers.",
            )

            if choice != "All practices":
                df_dash = df_dash[df_dash[practice_col].astype(str) == choice]

        if df_dash.empty:
            st.warning("No valid dates in the dataset after cleaning.")
        else:
            # 🔑 baseline = latest date in dataset
            # 🔑 baseline = latest date in dataset (shared)
            baseline = BASELINE_DATE
            st.caption(f"Baseline date (latest in your CSV): **{baseline.date()}**")

            # --- metric selector (simple for owner) ---
            default_metrics = [c for c in ["Collections", "Profit", "CollectionsPatient"] if c in df_dash.columns]
            if not default_metrics:
                numeric_cols = df_dash.select_dtypes(include="number").columns.tolist()
                numeric_cols = [c for c in numeric_cols if c not in ["dayofweek", "month", "year"]]
                default_metrics = numeric_cols[:1] if numeric_cols else []

            metric = st.selectbox(
                "What number do you want to track?",
                options=default_metrics,
                index=0 if default_metrics else 0,
                help="For example: Collections = total money collected.",
            )

            # --- time window relative to baseline ---
            window_label = st.selectbox(
                "Time window (counted backwards from the latest date in your data)",
                ["Last 30 days", "Last 90 days", "Last 365 days", "All history"],
            )

            window_days = {
                "Last 30 days": 30,
                "Last 90 days": 90,
                "Last 365 days": 365,
            }.get(window_label)

            if window_days is None:
                df_window = df_dash.copy()
            else:
                start = (baseline - pd.Timedelta(days=window_days)).normalize()
                end = baseline
                mask = (df_dash["date"] >= start) & (df_dash["date"] <= end)
                df_window = df_dash.loc[mask].copy()

            # previous window (for % change)
            prev_df = None
            if window_days is not None:
                prev_start = (baseline - pd.Timedelta(days=2 * window_days)).normalize()
                prev_end = (baseline - pd.Timedelta(days=window_days)).normalize()
                prev_mask = (df_dash["date"] >= prev_start) & (df_dash["date"] <= prev_end)
                prev_df = df_dash.loc[prev_mask].copy()

            # aggregate by date first
            if not df_window.empty:
                daily_window = df_window.groupby("date")[metric].sum()
                current_total = float(daily_window.sum())
            else:
                current_total = 0.0

            if not df_dash.empty:
                daily_all = df_dash.groupby("date")[metric].sum()
                all_total = float(daily_all.sum())
            else:
                all_total = 0.0

            if prev_df is not None and not prev_df.empty:
                daily_prev = prev_df.groupby("date")[metric].sum()
                prev_total = float(daily_prev.sum())
            else:
                prev_total = None

            # --- KPI cards (owner-friendly labels) ---
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    f"{metric} in this period",
                    f"${current_total:,.0f}",
                )
            with c2:
                if prev_total is not None and prev_total > 0:
                    delta_pct = (current_total - prev_total) / prev_total * 100
                    st.metric(
                        "Compare with previous same length period",
                        f"${prev_total:,.0f}",
                        f"{delta_pct:+.1f}%",
                    )
                else:
                    st.metric("Compare with previous period", "Not available", "—")
            with c3:
                st.metric(
                    f"{metric} over all data",
                    f"${all_total:,.0f}",
                )

            st.markdown("---")
            st.markdown(
                """
                ### How to read this

                - **Left chart**: How this metric changes over time.  
                - **Right chart**: Same number broken down by year and month.  
                - **Bottom charts**: Which **weekday**, **month**, or **year** is stronger or weaker.
                """
            )

            # --- charts ---
            tab_timeline, tab_drill = st.tabs(["Timeline view", "Breakdowns"])

            with tab_timeline:
                left, right = st.columns(2)
                with left:
                    st.plotly_chart(
                        timeseries_line(df_window, "date", metric, f"{metric} over time"),
                        use_container_width=True,
                    )
                with right:
                    st.plotly_chart(
                        year_month_breakdown(df_window, "date", metric, f"{metric} by Year/Month"),
                        use_container_width=True,
                    )

            with tab_drill:
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.plotly_chart(
                        bar_by(df_window, "dayofweek", metric, f"{metric} by weekday (0 = Monday)"),
                        use_container_width=True,
                    )
                if "year" in df_window.columns:
                    with b2:
                        st.plotly_chart(
                            bar_by(df_window, "year", metric, f"{metric} by year"),
                            use_container_width=True,
                        )
                if "month" in df_window.columns:
                    with b3:
                        st.plotly_chart(
                            bar_by(df_window, "month", metric, f"{metric} by month"),
                            use_container_width=True,
                        )

# ======================== 🧮 Collections Calculator ========================
elif mode == "🧮 Collections Calculator":
    st.subheader("Collections Mixer")
    st.markdown(
        """
        Build your **own collections metric** by mixing different columns.
        Great for testing “what counts as production?” before you lock it in.
        """
    )

    nums = df_feat.select_dtypes(include="number").columns.tolist()
    if not nums:
        st.warning("No numeric columns detected in the dataset.")
    else:
        with st.expander("Step 1 ▸ Choose ingredients", expanded=True):
            chosen = st.multiselect(
                "Pick the columns you want to include in your collections metric",
                options=nums,
                help="For example: patient portion, insurance portion, adjustments, etc.",
            )

        if not chosen:
            st.info("Select at least one column to get started.")
        else:
            col_left, col_right = st.columns([2, 1])

            with col_left:
                method = st.radio(
                    "How should we combine them?",
                    ["Sum", "Average", "Weighted mix"],
                    help=(
                        "Sum = add them up row by row.  \n"
                        "Average = take the mean per row.  \n"
                        "Weighted mix = let you emphasise some columns more than others."
                    ),
                )

            with col_right:
                new_name = st.text_input(
                    "Name of the new metric",
                    value="CollectionsCombined",
                    help="This will be the new column name in the data.",
                )

            weights = None
            if method == "Weighted mix":
                st.markdown("#### Step 2 ▸ Set weights (how important is each column?)")
                weight_vals = []
                for col in chosen:
                    w = st.slider(
                        f"Weight for `{col}`",
                        min_value=0.0,
                        max_value=5.0,
                        value=1.0,
                        step=0.1,
                    )
                    weight_vals.append(w)

                total_w = sum(weight_vals)
                if total_w == 0:
                    weights = [1.0 / len(weight_vals)] * len(weight_vals)
                    st.caption("All weights are 0, using equal weights instead.")
                else:
                    weights = [w / total_w for w in weight_vals]
                st.caption(
                    "Weights are normalised to add up to 1. "
                    "Higher weight = that column matters more in the mix."
                )

            set_primary = st.checkbox(
                "Use this as the main 'Collections' metric for the rest of the app",
                value=True,
            )

            if st.button("🔧 Build this collections metric"):
                try:
                    if combine_columns:
                        df2 = combine_columns(
                            df_feat,
                            chosen,
                            method="sum" if method == "Sum" else "avg" if method == "Average" else "weighted",
                            weights=weights,
                            new_name=new_name,
                        )
                    else:
                        # fallback implementation
                        df2 = df_feat.copy()
                        if method == "Sum":
                            df2[new_name] = df2[chosen].sum(axis=1)
                        elif method == "Average":
                            df2[new_name] = df2[chosen].mean(axis=1)
                        else:  # Weighted mix
                            ws = np.array(weights) if weights is not None else np.ones(len(chosen))
                            ws = ws / (ws.sum() if ws.sum() else 1.0)
                            df2[new_name] = (df2[chosen] * ws).sum(axis=1)

                    # --- Preview & download ---
                    st.success(f"Created column: `{new_name}`")

                    avg_val = float(df2[new_name].mean())
                    total_val = float(df2[new_name].sum())

                    k1, k2 = st.columns(2)
                    with k1:
                        st.metric("Average per row", f"${avg_val:,.0f}")
                    with k2:
                        st.metric("Total across all rows", f"${total_val:,.0f}")

                    st.markdown("#### Preview of the new metric (first 15 rows)")
                    st.dataframe(df2[[*chosen, new_name]].head(15), use_container_width=True)

                    # Small bar chart: how big each ingredient is vs the new mix
                    st.markdown("#### How big is each ingredient vs the new metric?")
                    avg_series = df2[chosen + [new_name]].mean().rename("Average per row")
                    st.bar_chart(avg_series)

                    st.download_button(
                        "⬇️ Download CSV with new metric",
                        df2.to_csv(index=False).encode("utf-8"),
                        file_name="dataset_with_collections_combined.csv",
                        mime="text/csv",
                    )

                    if set_primary:
                        df_feat[new_name] = df2[new_name]
                        df_feat["Collections"] = df2[new_name]
                        st.info(f"`{new_name}` is now the session 'Collections' metric used in other pages.")
                except Exception as e:
                    st.error(f"Error while building metric: {e}")

# ======================== 🧠 Strategies (LLM) ========================
elif mode == "🧠 Strategies (LLM)":
    st.subheader("No-Show & Revenue Strategy Lab")

    # If helper modules are missing, show a friendly message and stop
    if not (scrape_public_pages and draft_strategies):
        st.info(
            "Optional feature. Add `modules/scraper.py` and `modules/llm_helper.py` "
            "to enable AI-based strategy generation from public articles."
        )
    else:
        st.markdown(
            """
            Turn your **numbers + public articles** into a ready-to-use playbook
            for cutting no-shows and lifting production.
            """
        )

        # ---------- STEP 1: Goal & Style ----------
        st.markdown("### Step 1 ▸ Choose the goal and style")

        col_goal, col_style = st.columns(2)
        with col_goal:
            goal = st.selectbox(
                "Main goal for this plan",
                [
                    "Cut no-shows",
                    "Fill hygiene schedule",
                    "Grow high-value treatment (crowns / implants)",
                    "Fix problem day (e.g., Mondays)",
                    "Improve recall / reactivation",
                ],
            )

        with col_style:
            style = st.radio(
                "How should the plan read?",
                ["Quick checklist", "Detailed playbook", "Email to my team"],
                help=(
                    "Quick checklist = bullet list of actions.\n"
                    "Detailed playbook = longer, with examples.\n"
                    "Email to my team = friendly email you can paste into your inbox."
                ),
            )

        horizon = st.selectbox(
            "Time horizon",
            ["Next 2 weeks", "Next 1 month", "Next quarter"],
            help="How far ahead this plan should look.",
        )

        st.markdown("---")
        # ---------- STEP 2: Current numbers ----------
        st.markdown("### Step 2 ▸ Enter your current numbers")

        # Default no-show rate from your dataset (fallback = 0.20)
        default_no_show = 0.20
        if "target_no_show" in df_feat.columns:
            try:
                ns_series = pd.to_numeric(df_feat["target_no_show"], errors="coerce")
                if ns_series.notna().any():
                    default_no_show = float(ns_series.mean())
                    # clamp to [0, 1] so slider is safe
                    default_no_show = max(0.0, min(1.0, default_no_show))
            except Exception:
                default_no_show = 0.20

        c1, c2, c3 = st.columns(3)
        with c1:
            score = st.slider(
                "Average no-show rate (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=float(default_no_show),
                step=0.01,
                help="For example, 0.20 means 20% of appointments are missed.",
            )

        with c2:
            shortfall = st.number_input(
                "Daily collections shortfall ($)",
                min_value=0.0,
                value=800.0,
                step=50.0,
                help="How much below your target you usually are per day.",
            )

        with c3:
            open_slots = st.number_input(
                "Open slots vs target (per day)",
                min_value=0,
                value=3,
                step=1,
                help="How many more appointments you wish you could add each day.",
            )

        # Quick KPI cards
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("No-show rate", f"{score:.1%}")
        with k2:
            st.metric("Daily shortfall", f"${shortfall:,.0f}")
        with k3:
            st.metric("Open slots / day", f"{open_slots}")

        # --- Use your dataset to guess current $ per filled slot ---
        avg_coll = None
        if "Collections" in df_feat.columns:
            coll_series = pd.to_numeric(df_feat["Collections"], errors="coerce")
            if coll_series.notna().any():
                avg_coll = float(coll_series.mean())

        if avg_coll is None or np.isnan(avg_coll):
            avg_coll = 300.0  # safe default if data is messy

        work_days_per_month = 20

        # Capacity if you fill extra slots, using your current value per slot
        capacity_upside = open_slots * avg_coll * work_days_per_month

        # Gap implied by your "daily collections shortfall" slider
        gap_upside = shortfall * work_days_per_month

        # We cannot recover more than the smaller of capacity and gap
        estimated_monthly_upside = min(capacity_upside, gap_upside)

        # How much $ per extra slot you would need to fully close the daily gap
        if open_slots > 0:
            required_per_slot = shortfall / open_slots
        else:
            required_per_slot = avg_coll

        st.markdown("#### Snapshot of the opportunity")
        s1, s2 = st.columns(2)
        with s1:
            st.metric(
                "Current est. revenue per filled slot (from data)",
                f"${avg_coll:,.0f}",
            )
        with s2:
            st.metric(
                "Needed $ per extra slot to fully close the daily gap",
                f"${required_per_slot:,.0f}",
                help="Daily shortfall ÷ open slots per day.",
            )

        st.markdown("Small visual: today's gap vs opportunity")
        gap_df = pd.DataFrame(
            {
                "Metric": ["Daily shortfall (per day)", "Potential monthly upside"],
                "Value": [shortfall, estimated_monthly_upside],
            }
        ).set_index("Metric")
        st.bar_chart(gap_df)


        # ---------- STEP 3: Public articles (optional) ----------
        st.markdown("### Step 3 ▸ Add public URLs for inspiration (optional)")
        urls_text = st.text_area(
            "One public article per line",
            height=120,
            placeholder=(
                "https://www.dentistryiq.com/... \n"
                "https://www.ada.org/... \n"
                "https://some-marketing-blog.com/..."
            ),
        )
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]

        st.caption(
            "The app will read these pages and ask the AI to adapt the best ideas "
            "to your practice, goal, and numbers."
        )

        st.markdown("---")

        # ---------- Automated insights BEFORE AI ----------
        st.markdown("### Practice snapshot (before AI)")

        insights = []

        # No-show assessment
        if score < 0.08:
            ns_text = "Your no-show rate is **better than typical** (often 10–12%)."
        elif score < 0.15:
            ns_text = (
                "Your no-show rate is in the **typical range** for many practices, "
                "but still worth improving."
            )
        else:
            ns_text = (
                "Your no-show rate is **higher than usual** (above ~15%). "
                "This should be a priority to fix."
            )
        insights.append(ns_text)

        # Shortfall assessment
        if shortfall == 0:
            sf_text = "You did not report a daily collections shortfall."
        elif shortfall < 500:
            sf_text = "You are **a little below** your daily collections target."
        else:
            sf_text = "You are **significantly below** your daily collections target."
        insights.append(sf_text)

        # Open slots assessment
        if open_slots == 0:
            os_text = "You reported **no open slots** vs target. Utilisation looks good."
        elif open_slots <= 2:
            os_text = "You have a **small gap** in daily slots. Tighten scheduling and confirmations."
        else:
            os_text = "You have a **large number of open slots** each day. This is a big opportunity."
        insights.append(os_text)

        st.markdown("**Automated read of your situation:**")
        for txt in insights:
            st.markdown(f"- {txt}")

        st.markdown(
            f"- If you fill those extra slots, you could recover about **${estimated_monthly_upside:,.0f} per month**."
        )

        # Bundle everything into a payload for the LLM helper
        kpi = {
            "no_show_risk_avg": score,
            "predicted_collections_shortfall": shortfall,
            "forecast_open_slots_gap": open_slots,
            "goal": goal,
            "style": style,
            "time_horizon": horizon,
            "avg_collections_per_slot": avg_coll,
            "estimated_monthly_upside": estimated_monthly_upside,
            "insights": insights,
        }

        st.markdown("---")

        # ---------- Generate strategy with AI ----------
        if st.button("🧠 Generate strategy playbook", type="primary", key="generate_strategy"):
            st.write("⬇️ Button clicked – generating your playbook...")

            try:
                # 1) Scrape public pages (if any)
                if urls:
                    with st.spinner("Scraping public pages for ideas..."):
                        scraped = scrape_public_pages(urls, max_pages=5, max_chars=4000)
                else:
                    scraped = ""

                # 2) Call the LLM helper
                with st.spinner("Asking AI to draft your playbook..."):
                    plan = draft_strategies(kpi, scraped)

                # 3) Show the result
                st.success("Strategy playbook generated below.")
                st.markdown("### Recommended Strategy Playbook")
                st.write(plan)

                st.markdown("#### TL;DR – copy these into your slide or email")
                st.info(
                    "Pick the top 3–5 actions from the playbook above and paste them "
                    "here as a short checklist for your team."
                )

                st.download_button(
                    "⬇️ Download strategy_playbook.txt",
                    plan.encode("utf-8"),
                    file_name="strategy_playbook.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"Error while generating strategy: {e}")

# ======================== ❓ Data Q&A ========================
else:  # "❓ Data Q&A"
    st.subheader("Ask questions about your data")

    if "date" in df_feat.columns:
        st.caption(
            f"For time questions, treat **{BASELINE_DATE.date()}** "
            f"as 'today' (latest date in your dataset)."
        )

    if not (build_context_cards and chat_answer):
        st.info("Optional: add `modules/chatbot.py` if you want a local LLM Q&A over your dataset.")
    else:
        # (Optional) pass baseline into your chatbot if you update its signature
        # ctx = build_context_cards(df_feat, baseline_date=BASELINE_DATE)
        ctx = build_context_cards(df_feat)

        q = st.text_input("Question (e.g., 'What was collections last month?')")
        if st.button("Ask") and q.strip():
            try:
                # ans = chat_answer(q, df_feat, ctx, baseline_date=BASELINE_DATE)
                ans = chat_answer(q, df_feat, ctx)
                st.write(ans)
            except Exception as e:
                st.error(f"Chat error: {e}")
