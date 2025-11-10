from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)
from sklearn.utils import resample


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "Unnamed: 0",
        "first",
        "last",
        "street",
        "trans_num",
        "zip",
        "city",
        "unix_time",
        "job",
        "cc_num",
    ]

    df = df.drop(
        columns=[col for col in drop_cols if col in df.columns], errors="ignore"
    )
    return df


def extract_time_and_age_features(df: pd.DataFrame) -> pd.DataFrame:
    if "trans_date_trans_time" not in df.columns or "dob" not in df.columns:
        raise ValueError(
            "Expected columns 'trans_date_trans_time' and 'dob' not found in DataFrame."
        )

    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], errors="coerce"
    )
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["weekday"] = df["trans_date_trans_time"].dt.weekday  # Monday=0
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year

    median_age = df["age"].median(skipna=True)
    df["age"] = df["age"].apply(
        lambda x: median_age if pd.isna(x) or x < 10 or x > 100 else x
    )

    df = df.drop(columns=["trans_date_trans_time", "dob"], errors="ignore")

    return df


def extract_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["lat", "long", "merch_lat", "merch_long"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # Convert to radians for haversine formula
    lat1 = np.radians(df["lat"])
    lon1 = np.radians(df["long"])
    lat2 = np.radians(df["merch_lat"])
    lon2 = np.radians(df["merch_long"])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371  # Earth radius in km

    df["distance_from_home"] = R * c

    median_dist = df["distance_from_home"].median(skipna=True)
    df["distance_from_home"] = df["distance_from_home"].apply(
        lambda x: median_dist if pd.isna(x) or x < 0 or x > 20000 else x
    )

    df = df.drop(columns=required_cols, errors="ignore")

    return df


def sklearn_basic_encode(
    df: pd.DataFrame, fit: bool = True, encoders: dict | None = None, mode: str = "tree"
):
    df_out = df.copy()
    if encoders is None:
        encoders = {}

    if "merchant" in df_out.columns:
        if fit:
            merchant_enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            merchant_enc.fit(df_out[["merchant"]].astype(str))
            encoders["merchant_enc"] = merchant_enc
        else:
            merchant_enc = encoders["merchant_enc"]
        df_out["merchant"] = merchant_enc.transform(
            df_out[["merchant"]].astype(str)
        ).astype(int)

    if "state" in df_out.columns:
        if fit:
            state_enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            state_enc.fit(df_out[["state"]].astype(str))
            encoders["state_enc"] = state_enc
        else:
            state_enc = encoders["state_enc"]
        df_out["state"] = state_enc.transform(df_out[["state"]].astype(str)).astype(int)

    if "gender" in df_out.columns:
        if fit:
            gender_enc = LabelEncoder()
            gender_enc.fit(df_out["gender"].astype(str))
            encoders["gender_enc"] = gender_enc
        else:
            gender_enc = encoders["gender_enc"]
        df_out["gender"] = gender_enc.transform(df_out["gender"].astype(str)).astype(
            int
        )

    if "category" in df_out.columns:
        if mode == "tree":
            if fit:
                cat_enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                cat_enc.fit(df_out[["category"]].astype(str))
                encoders["category_enc"] = cat_enc
            else:
                cat_enc = encoders["category_enc"]
            df_out["category"] = cat_enc.transform(
                df_out[["category"]].astype(str)
            ).astype(int)
        else:
            if fit:
                cat_enc = OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore", drop="first"
                )
                cat_enc.fit(df_out[["category"]].astype(str))
                encoders["category_enc"] = cat_enc
            else:
                cat_enc = encoders["category_enc"]
            cat_ohe = cat_enc.transform(df_out[["category"]].astype(str))
            ohe_cols = cat_enc.get_feature_names_out(["category"])
            df_ohe = pd.DataFrame(cat_ohe, columns=ohe_cols, index=df_out.index)
            df_out = pd.concat([df_out.drop(columns=["category"]), df_ohe], axis=1)

    return df_out, encoders


def scale_numeric(df, fit=True, scalers=None, mode="linear"):
    df_out = df.copy()
    if scalers is None:
        scalers = {}

    cols_to_scale = ["amt", "city_pop", "age", "distance_from_home"]

    for col in ["amt", "city_pop", "distance_from_home"]:
        df_out[col] = np.log1p(df_out[col].clip(lower=0))

    if fit:
        scaler = StandardScaler() if mode == "linear" else MinMaxScaler()
        scaler.fit(df_out[cols_to_scale])
        scalers["numeric_scaler"] = scaler
    else:
        scaler = scalers["numeric_scaler"]

    df_out[cols_to_scale] = scaler.transform(df_out[cols_to_scale])
    return df_out, scalers


def resample_fraud_data(df: pd.DataFrame, ratio: float, random_state: int = 42):
    if ratio <= 0:
        raise ValueError("ratio must be > 0 (e.g. 1.0 for 1:1, 0.1 for 1:10)")

    fraud = df[df["is_fraud"] == 1].copy()
    legit = df[df["is_fraud"] == 0].copy()

    n_fraud = len(fraud)
    n_legit = len(legit)

    desired_fraud = int(round(ratio * n_legit))
    desired_fraud = max(desired_fraud, 1)

    if desired_fraud <= n_fraud:
        fraud_upsampled = fraud.sample(
            n=desired_fraud, replace=False, random_state=random_state
        )
    else:
        fraud_upsampled = resample(
            fraud, replace=True, n_samples=desired_fraud, random_state=random_state
        )

    fraud_upsampled = pd.DataFrame(fraud_upsampled)
    legit = pd.DataFrame(legit)
    df_up = pd.concat([legit, fraud_upsampled], axis=0, ignore_index=True)
    df_up = df_up.sample(frac=1, random_state=random_state).reset_index(drop=True)

    desired_legit = int(round(n_fraud / ratio))
    desired_legit = max(desired_legit, 1)

    if desired_legit >= n_legit:
        legit_downsampled = legit.sample(
            n=n_legit, replace=False, random_state=random_state
        )
    else:
        legit_downsampled = resample(
            legit, replace=False, n_samples=desired_legit, random_state=random_state
        )

    legit_downsampled = pd.DataFrame(legit_downsampled)
    fraud = pd.DataFrame(fraud)
    df_down = pd.concat([fraud, legit_downsampled], axis=0, ignore_index=True)
    df_down = df_down.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_up, df_down


def prepare_data(
    df: pd.DataFrame,
    mode: str = "linear",
    training: bool = True,
    ratio: Optional[float] = 0.3,
    fit: bool = True,
    encoders: dict | None = None,
    scalers: dict | None = None,
):
    """
    Preprocesses the raw fraud dataset for model training or inference.

    Steps:
    1. Drops unnecessary columns.
    2. Extracts temporal (hour, day, month, weekday, weekend) and age features.
    3. Computes 'distance_from_home' from lat/long pairs.
    4. Encodes categorical variables (mode-aware).
    5. Scales numeric features (mode-aware).
    6. Optionally resamples for class balance during training.

    Args:
        df: Raw dataframe (train or test).
        mode: One of ["linear", "distance", "tree"].
              - "linear"   → StandardScaler + one-hot for 'category'
              - "distance" → MinMaxScaler + one-hot for 'category'
              - "tree"     → No scaling, ordinal encodings
        training: If True, performs up/down sampling of 'is_fraud'.
        ratio: Target fraud:legit ratio for resampling (e.g. 0.3 = 1:3.3).
        fit: If True, fit encoders/scalers. If False, reuse provided ones.
        encoders: Prefitted encoders dict (from training phase).
        scalers: Prefitted scalers dict (from training phase).

    Returns:
        dict with:
            {
              "df": preprocessed dataframe,
              "encoders": fitted or reused encoders,
              "scalers": fitted or reused scalers,
              "df_up": upsampled dataframe (if training),
              "df_down": downsampled dataframe (if training)
            }
    """
    df = drop_unwanted_columns(df)
    df = extract_time_and_age_features(df)
    df = extract_distance_feature(df)

    df, encoders_out = sklearn_basic_encode(
        df=df, fit=fit, encoders=encoders, mode=mode
    )

    scalers_out = {}
    if mode != "tree":
        df, scalers_out = scale_numeric(df, fit=fit, scalers=scalers, mode=mode)

    outputs = {"df": df, "encoders": encoders_out, "scalers": scalers_out}

    if training:
        if ratio is None:
            # Skip resampling, just return base df
            outputs["df_up"] = df.copy()
            outputs["df_down"] = df.copy()
        else:
            df_up, df_down = resample_fraud_data(df, ratio=ratio)
            outputs["df_up"] = df_up
            outputs["df_down"] = df_down

    return outputs
