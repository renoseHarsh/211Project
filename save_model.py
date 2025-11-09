# ==============================================================
# SAVE_MODEL.PY
# Run this file ONCE to train and save your best model
# and the encoders needed for the demo.
# ==============================================================
import joblib  # For saving the model and encoders
import numpy as np
import pandas as pd
import xgboost as xgb

from preprocessing import prepare_data

print("Starting model training and saving process...")

# ---
# 1. Load Full Training Data
# ---
train_df = pd.read_csv("fraudTrain.csv")
print("Loaded fraudTrain.csv")

# ---
# 2. Fit and Save the "Tree" Encoders
# ---
# We fit the encoders on the *entire* original training data
# so they know all possible categories.
out_train_init = prepare_data(
    train_df,
    mode="tree",
    training=False,  # We only need the encoders
    fit=True,
)
encoders = out_train_init["encoders"]
joblib.dump(encoders, "tree_encoders.joblib")
print(f"Encoders saved to 'tree_encoders.joblib'")

# ---
# 3. Create the Champion Training Set
# ---
# This is our winning combination:
# - mode="tree" (no scaling)
# - fit=False (use the encoders we just saved)
# - ratio=0.1 (the winning ratio)
# - training=True (to get the resampled data)
out_train_final = prepare_data(
    train_df,
    mode="tree",
    training=True,
    ratio=0.1,
    fit=False,
    encoders=encoders,
    scalers={},
)

# We use df_up, our winning sampling strategy
df_train = out_train_final["df_up"]
X_train = df_train.drop("is_fraud", axis=1)
y_train = df_train["is_fraud"]

# Clean inf/-inf values
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)

print(f"Created final training set (df_up, ratio=0.1). Shape: {X_train.shape}")

# ---
# 4
