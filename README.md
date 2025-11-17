# Credit Card Fraud Detection Project

This is a machine learning project for **CSET 211 Statistical machine learning** to identify fraudulent credit card transactions from a highly imbalanced dataset.

The project systematically compares **7 different classification models**, tests multiple **data-preprocessing** and **resampling strategies**, and identifies a **champion model (XGBoost)** capable of achieving a high **F1-Score**.

---

## Live Link

https://demopy-djqub5aee62lfeni55psmo.streamlit.app/

---

## Dataset Instructions (IMPORTANT)

The dataset is not included due to size. Download fraudTrain.csv and fraudTest.csv from the original source

## ⚙️ How to Run the Demo

The **demo** is a **Streamlit web app** that simulates transactions and returns live predictions using the trained XGBoost model.

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2: Train and Save the Model

Run the following command to train and save the final model + encoders:

```bash
python save_model.py
```

This will generate:

```
xgb_model.joblib
tree_encoders.joblib
```

---

### Step 3: Launch the Streamlit App

```bash
streamlit run demo_app.py
```

---

### Step 4: Open in Browser

Streamlit will display a local URL (for example):

```
http://localhost:8501
```

Open it in your browser to use the fraud detection demo.

---

## 🧩 Features of the Demo

- User inputs transaction data (amount, age, distance, etc.)
- App encodes and preprocesses data using saved encoders.
- XGBoost model predicts **Fraudulent (1)** or **Legitimate (0)**.
- Real-time prediction results displayed in the UI.

---

## 📚 References

- scikit-learn Documentation – https://scikit-learn.org
- XGBoost: Chen & Guestrin (2016) “A Scalable Tree Boosting System” – KDD 2016
- Dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection

---
