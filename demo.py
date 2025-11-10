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
# Use Streamlit's cache to load these only once, speeding up the app.
@st.cache_resource
def load_model_and_encoders():
    """
    Loads the champion model and the preprocessing encoders.
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
    # Stop the app if artifacts failed to load
    st.stop()


# =C==================================================================
# 2. DEFINE APP UI
# ====================================================================

st.set_page_config(page_title="Fraud Model Calculator", layout="wide")
st.title("💳 XGBoost Fraud Model Calculator")
st.markdown(f"Using **XGBoost** (F1: **0.854**)")

# --- Dynamically get options from the loaded encoders ---
try:
    CATEGORIES = list(encoders["category_enc"].categories_[0])
    MERCHANTS = list(encoders["merchant_enc"].categories_[0])
    GENDERS = list(encoders["gender_enc"].classes_)  # 'classes_' for LabelEncoder
    STATES = list(encoders["state_enc"].categories_[0])
except Exception as e:
    st.error(f"FATAL: Error reading categories from loaded 'encoders.joblib'.")
    st.error(f"Details: {e}")
    st.stop()  # Can't build UI if options are missing


# --- Create Input Form ---
# We use a form so all inputs are submitted at once.
with st.form("transaction_form"):
    st.header("Transaction Inputs")

    col1, col2, col3 = st.columns(3)

    # --- COLUMN 1: Categorical Inputs ---
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
            "Merchant", options=MERCHANTS, index=0  # Default to the first merchant
        )

        selected_gender = st.selectbox(
            "Customer Gender", options=GENDERS, index=0  # Default to the first gender
        )

        selected_state = st.selectbox(
            "State (Abbr.)",
            options=STATES,
            index=STATES.index("PA") if "PA" in STATES else 0,
        )

    # --- COLUMN 2: Numerical "Expert" Inputs ---
    with col2:
        st.subheader("Numerical Features")

        # We use your calculated defaults here
        amt = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            max_value=1000000.0,
            value=47.52,  # Your default
            step=50.0,
        )

        city_pop = st.number_input(
            "City Population",
            min_value=0,
            max_value=40_000_000,  # Max ~Tokyo
            value=2456,  # Your default
            step=1000,
        )

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=44,  # Your default
            step=1,
        )

    # --- COLUMN 3: Time & Key Fraud Knobs ---
    with col3:
        st.subheader("Engineered Features")

        # Smart Date/Time Input
        trans_date = st.date_input("Transaction Date")
        trans_time = st.time_input("Transaction Time")

        # The "Fraud Knob"
        distance_from_home = st.slider(
            "Distance From Home (km)",
            min_value=0.0,
            max_value=20000.0,  # ~Half Earth's circumference
            value=78.23,  # Your default
            step=10.0,
        )

    # --- Submit Button ---
    st.divider()
    submitted = st.form_submit_button(
        "Analyze Transaction", type="primary", use_container_width=True
    )

# ====================================================================
# 3. TRANSFORM 9 INPUTS INTO 13 MODEL FEATURES
# ====================================================================

# --- 3a. Engineer Time Features (5 features from 1 input) ---

# Combine date and time from UI into a single datetime object
transaction_datetime = datetime.combine(trans_date, trans_time)

# Extract the 5 time-based features
hour = transaction_datetime.hour
day = transaction_datetime.day
month = transaction_datetime.month
weekday = transaction_datetime.weekday()  # Monday=0, Sunday=6
is_weekend = 1 if weekday >= 5 else 0

# --- 3b. Encode Categorical Features (4 features from 4 inputs) ---

# We create a mini-DataFrame for each input to silence the UserWarning.
# This provides the feature name the encoder expects.
try:
    category_df = pd.DataFrame([selected_category], columns=["category"])
    category_encoded = encoders["category_enc"].transform(category_df)[0][0]

    merchant_df = pd.DataFrame([selected_merchant], columns=["merchant"])
    merchant_encoded = encoders["merchant_enc"].transform(merchant_df)[0][0]

    state_df = pd.DataFrame([selected_state], columns=["state"])
    state_encoded = encoders["state_enc"].transform(state_df)[0][0]

    # LabelEncoder doesn't need feature names, so this is fine.
    gender_encoded = encoders["gender_enc"].transform([selected_gender])[0]

except Exception as e:
    st.error(f"FATAL: An error occurred during feature encoding.")
    st.error(f"Details: {e}")
    st.stop()

# --- 3c. Assemble the Final 13-Feature Vector ---

# Get the 4 "expert" inputs directly from the UI
# (amt, city_pop, age, distance_from_home)

# Create a dictionary for all 13 features
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

# Create a single-row DataFrame
X_processed = pd.DataFrame(feature_data, index=[0])

# --- 3d. FINAL STEP: Ensure Column Order ---

# Get the exact feature order the model was trained on
expected_features = model.feature_names_in_

# Re-order our DataFrame to match the model's expectation.
# This prevents the "feature_names mismatch" error.
X_processed = X_processed[expected_features]


# ====================================================================
# 4. RUN PREDICTION & DISPLAY RESULTS
# ====================================================================

try:
    # --- 4a. Get predictions from the champion model ---

    # .predict() gives the final class (0 or 1)
    prediction = model.predict(X_processed)[0]

    # .predict_proba() gives the % probability for each class
    # We want the probability of class '1' (Fraud), which is at index [0][1]
    probability = model.predict_proba(X_processed)[0][1]

    # --- 4b. Display the result ---

    st.header("Analysis Result")

    if prediction == 1:
        # If the model predicts Fraud (1)
        st.error(f"**FRAUD DETECTED** (Confidence: {probability:.2%})")
        st.markdown(
            f"The model is **{probability:.2%}** confident this is a fraudulent transaction."
        )

    else:
        # If the model predicts Normal (0)
        st.success(f"**Transaction Normal** (Fraud Probability: {probability:.2%})")
        st.markdown(
            f"The model found a low **{probability:.2%}** probability of fraud."
        )

    # --- 4c. Show the "work" ---

    # This expander is great for your professor
    with st.expander("Show Final 13-Feature Vector (Data Sent to Model)"):
        st.dataframe(X_processed)

except Exception as e:
    st.error("An error occurred during the prediction phase.")
    st.error(f"Details: {e}")
