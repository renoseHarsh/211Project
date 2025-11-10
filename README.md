# Credit Card Fraud Detection Project

This is a machine learning project for **[Your Class Name]** to identify fraudulent credit card transactions from a highly imbalanced dataset.

The project systematically compares **7 different classification models**, tests multiple **data-preprocessing** and **resampling strategies**, and identifies a **champion model (XGBoost)** capable of achieving a high **F1-Score**.

---

## Live Link
https://renoseharsh-211project-demo-dfek0s.streamlit.app/

---

## 🧠 Executive Summary: The Gauntlet

The primary evaluation metric for this **highly imbalanced classification problem** is the **F1-Score**.

### 1. The "Trash Tier" (F1 < 0.05)
- **Models:** KNN, Naive Bayes  
- **Reason for Failure:**  
  - KNN suffered from the *Curse of Dimensionality* (25 features → meaningless distance metrics).  
  - Naive Bayes failed due to **false independence assumptions**.  
- **Files:** [cite: `knn_results.csv`, `naive_bayes_results.csv`]

### 2. The Baseline (F1 ≈ 0.37)
- **Models:** Logistic Regression, Linear SVM  
- **Summary:**  
  Simple linear decision boundaries, decent precision but poor recall.  
- **Files:** [cite: `logreg_results.csv`, `svm_results.csv`]

### 3. The Overfitting Lesson (F1 ≈ 0.56)
- **Model:** Decision Tree Classifier  
- **Observation:**  
  - Unrestricted trees memorized the dataset (overfitting).  
  - Pruning (`max_depth=10`) improved generalization.  
- **Files:** [cite: `dtc_results.csv`]

### 4. The Ensemble Fix (F1 ≈ 0.83)
- **Model:** Random Forest  
- **Mechanism:**  
  Bagging multiple trees reduced variance and stabilized predictions.  
- **Files:** [cite: `rf_results.csv`]

### 5. The Champion (F1 ≈ 0.85)
- **Model:** XGBoost  
- **Mechanism:**  
  Boosting sequentially corrects errors of prior trees.  
- **Result:** Achieved the highest F1-Score in the gauntlet.  
- **Files:** [cite: `xgb_results.csv`]

---

## 🔍 Key Strategic Findings

### **1. Resampling Ratio**
- A **1:1 (fraud:legit)** ratio was **a failed strategy**.
- Models began to “cry wolf” — very high recall, terrible precision.
- The **optimal ratio was 0.1 (1 fraud : 10 legit)**.
- **Reference:** [cite: `rf_results.csv`]

**Conclusion:**  
The sweet spot between data balance and realism was **0.1** — not fully balanced, but informative enough.

---

### **2. Sampling Method**
- For complex ensembles (**Random Forest, XGBoost**):
  - **Upsampling (`df_up`)** → F1 = **0.85**
  - **Downsampling (`df_down`)** → F1 = **0.63**
- **Reason:** Upsampling provides more learning material for the minority class.  
- **Reference:** [cite: `xgb_results.csv`]

**Conclusion:**  
Upsampling worked best for tree-based models by providing more varied minority samples.

---

## 🏆 Champion Model

| Property | Value |
|-----------|--------|
| **Model** | XGBoost |
| **Strategy** | Upsampling (`df_up`) |
| **Resampling Ratio** | 0.1 (1 fraud : 10 legit) |
| **Final F1-Score** | **0.851** |
| **Precision** | **0.909** |
| **Recall** | **0.800** |

**Summary:**  
XGBoost + Controlled Upsampling (ratio=0.1) achieved the perfect tradeoff between catching frauds and minimizing false alarms.

---

## ⚙️ How to Run the Demo

The **demo** is a **Streamlit web app** that simulates transactions and returns live predictions using the trained XGBoost model.

### Step 1: Install Dependencies
```bash
pip install pandas numpy xgboost scikit-learn joblib streamlit
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
- Dataset: *fraudTrain.csv* (Kaggle – Credit Card Fraud Detection Challenge)

---

## 💡 Final Takeaway
> “Fraud detection isn’t about accuracy — it’s about balance.  
> Once the data is balanced, boosting makes the imbalance irrelevant.”

