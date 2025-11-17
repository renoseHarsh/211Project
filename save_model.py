import time

import joblib
import pandas as pd
import xgboost as xgb

from preprocessing import prepare_data

# ====================================================================
# SCRIPT CONFIGURATION
# ====================================================================

# --- Champion Model Configuration ---
CHAMPION_CONFIG = {
    "model_params": {
        "name": "XGB_Resampled",
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "random_state": 42,
        },
    },
    "preprocessing_params": {
        "mode": "tree",
        "ratio": 0.1,
        "resample_type": "df_up",
    },
}


# --- File Paths ---
TRAIN_DATA_PATH = "fraudTrain.csv"
MODEL_SAVE_PATH = "models/xgb_champion.json"
ENCODERS_SAVE_PATH = "models/encoders.joblib"

print(f"Starting model production script...")
print(f"  Champion: XGBoost (Upsampled, ratio=0.1)")
print("=" * 60)

# ====================================================================
# 1. LOAD DATA
# ====================================================================
start_time = time.time()
print(f"Loading training data from {TRAIN_DATA_PATH}...")
try:
    train_df = pd.read_csv(TRAIN_DATA_PATH)
except FileNotFoundError:
    print(f"Error: {TRAIN_DATA_PATH} not found.")
    exit()

print(f"  Loaded {len(train_df)} initial records.")


# ====================================================================
# 2. PREPROCESSING (FIT ENCODERS & PREPARE DATA)
# ====================================================================
print("\nRunning preprocessing...")

# Step 2a: Fit encoders
print("  Fitting encoders on original data...")
out_init = prepare_data(
    train_df,
    mode=CHAMPION_CONFIG["preprocessing_params"]["mode"],
    training=False,  # We are not training yet
    fit=True
)
encoders = out_init["encoders"]
print("  Encoders fitted.")

# Step 2b: Get the final, resampled training data
print(f"  Preparing final training data (mode='tree', ratio=0.1, upsampling)...")
out_train = prepare_data(
    train_df,
    mode=CHAMPION_CONFIG["preprocessing_params"]["mode"],
    training=True,  # Now we get the training sets
    ratio=CHAMPION_CONFIG["preprocessing_params"]["ratio"],
    fit=False,  # We already fitted, just transform
    encoders=encoders,
    scalers={},  # Not used in 'tree' mode
)

df_train_final = out_train[CHAMPION_CONFIG["preprocessing_params"]["resample_type"]]
X_train = df_train_final.drop("is_fraud", axis=1)
y_train = df_train_final["is_fraud"]

print(f"  Final training data shape: {X_train.shape}")


# ====================================================================
# 3. TRAIN CHAMPION MODEL
# ====================================================================
print("\nTraining champion XGBoost model...")
print(f"  Parameters: {CHAMPION_CONFIG['model_params']}")

model = xgb.XGBClassifier(**CHAMPION_CONFIG["model_params"])

model.fit(X_train, y_train)

print("  Model training complete.")


# ====================================================================
# 4. SAVE ARTIFACTS
# ====================================================================
print("\nSaving model and encoders to /models directory...")

# Save the XGBoost model (use .json, it's safer than pickle)
model.save_model(MODEL_SAVE_PATH)
print(f"  ✅ Model saved to: {MODEL_SAVE_PATH}")

# Save the encoders
joblib.dump(encoders, ENCODERS_SAVE_PATH)
print(f"  ✅ Encoders saved to: {ENCODERS_SAVE_PATH}")


# ====================================================================
# 5. COMPLETE
# ====================================================================
end_time = time.time()
print("=" * 60)
print(f"Production script complete in {end_time - start_time:.2f} seconds.")
print("You are now ready to build the Streamlit demo.")
