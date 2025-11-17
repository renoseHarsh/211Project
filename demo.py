from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from preprocessing import prepare_data


# ====================================================================
# 1. LOAD ARTIFACTS (RUNS ONCE & CACHED)
# ====================================================================
@st.cache_resource
def load_model_and_encoders():
    """
    Loads the best model and the preprocessing encoders.
    """
    try:
        model = xgb.XGBClassifier()
        model.load_model("models/xgb_champion.json")
    except Exception as e:
        st.error(f"FATAL: Could not load model. Did you run 'save_model.py'?")
        st.error(f"Details: {e}")
        return None, None

    try:
        encoders = joblib.load("models/encoders.joblib")
    except Exception as e:
        st.error(f"FATAL: Could not load encoders. Did you run 'save_model.py'?")
        st.error(f"Details: {e}")
        return None, None

    return model, encoders


# --- Load Artifacts ---
model, encoders = load_model_and_encoders()
if model is None or encoders is None:
    st.stop()


# =C==================================================================
# 2. DEFINE APP UI
# ====================================================================

st.set_page_config(page_title="Fraud Model Calculator", layout="wide")
st.title("💳 XGBoost Fraud Model Calculator")
st.markdown(f"Using **XGBoost** (F1: **0.854**)")

try:
    CATEGORIES = list(encoders["category_enc"].categories_[0])
    MERCHANTS = list(encoders["merchant_enc"].categories_[0])
    GENDERS = list(encoders["gender_enc"].classes_)
    STATES = list(encoders["state_enc"].categories_[0])
except Exception as e:
    st.error(f"FATAL: Error reading categories from loaded 'encoders.joblib'.")
    st.error(f"Details: {e}")
    st.stop()


# --- Create Input Form ---
with st.form("transaction_form"):
    st.header("Transaction Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Categorical Features")
        selected_category = st.selectbox(
            "Category",
            options=CATEGORIES,
            index=(
                CATEGORIES.index("gas_transport")
                if "gas_transport" in CATEGORIES
                else 0
            ),
        )

        selected_merchant = st.selectbox(
            "Merchant", options=MERCHANTS, index=0
        )

        selected_gender = st.selectbox(
            "Customer Gender", options=GENDERS, index=0
        )

        selected_state = st.selectbox(
            "State (Abbr.)",
            options=STATES,
            index=STATES.index("PA") if "PA" in STATES else 0,
        )

    with col2:
        st.subheader("Numerical Features")

        amt = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            max_value=1000000.0,
            value=47.52,
            step=50.0,
        )

        city_pop = st.number_input(
            "City Population",
            min_value=0,
            max_value=40_000_000,
            value=2456,
            step=1000,
        )

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=44,
            step=1,
        )

    with col3:
        st.subheader("Engineered Features")

        trans_date = st.date_input("Transaction Date")
        trans_time = st.time_input("Transaction Time")

        distance_from_home = st.slider(
            "Distance From Home (km)",
            min_value=0.0,
            max_value=20000.0,
            value=78.23,
            step=10.0,
        )

    st.divider()
    submitted = st.form_submit_button(
        "Analyze Transaction", type="primary", use_container_width=True
    )

# ====================================================================
# 3. TRANSFORM 9 INPUTS INTO 13 MODEL FEATURES
# ====================================================================


transaction_datetime = datetime.combine(trans_date, trans_time)

hour = transaction_datetime.hour
day = transaction_datetime.day
month = transaction_datetime.month
weekday = transaction_datetime.weekday()  # Monday=0, Sunday=6
is_weekend = 1 if weekday >= 5 else 0

try:
    category_df = pd.DataFrame([selected_category], columns=["category"])
    category_encoded = encoders["category_enc"].transform(category_df)[0][0]

    merchant_df = pd.DataFrame([selected_merchant], columns=["merchant"])
    merchant_encoded = encoders["merchant_enc"].transform(merchant_df)[0][0]

    state_df = pd.DataFrame([selected_state], columns=["state"])
    state_encoded = encoders["state_enc"].transform(state_df)[0][0]

    gender_encoded = encoders["gender_enc"].transform([selected_gender])[0]

except Exception as e:
    st.error(f"FATAL: An error occurred during feature encoding.")
    st.error(f"Details: {e}")
    st.stop()

feature_data = {
    "merchant": merchant_encoded,
    "category": category_encoded,
    "amt": amt,
    "gender": gender_encoded,
    "state": state_encoded,
    "city_pop": city_pop,
    "hour": hour,
    "day": day,
    "month": month,
    "weekday": weekday,
    "is_weekend": is_weekend,
    "age": age,
    "distance_from_home": distance_from_home,
}

X_processed = pd.DataFrame(feature_data, index=[0])

expected_features = model.feature_names_in_

X_processed = X_processed[expected_features]


# ====================================================================
# 4. RUN PREDICTION & DISPLAY RESULTS
# ====================================================================

try:
    prediction = model.predict(X_processed)[0]

    probability = model.predict_proba(X_processed)[0][1]

    st.header("Analysis Result")

    if prediction == 1:
        st.error(f"**FRAUD DETECTED** (Confidence: {probability:.2%})")
        st.markdown(
            f"The model is **{probability:.2%}** confident this is a fraudulent transaction."
        )

    else:
        st.success(f"**Transaction Normal** (Fraud Probability: {probability:.2%})")
        st.markdown(
            f"The model found a low **{probability:.2%}** probability of fraud."
        )

    with st.expander("Show Final 13-Feature Vector (Data Sent to Model)"):
        st.dataframe(X_processed)

except Exception as e:
    st.error("An error occurred during the prediction phase.")
    st.error(f"Details: {e}")
