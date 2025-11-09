# From Baseline to Boosting: Tackling Class Imbalance in Fraud Detection

---

## **Page 1: Introduction & The Problem of Imbalance**

### **Dataset Overview**
The dataset used for this project is **`fraudTrain.csv`**, which contains anonymized credit card transaction data. Each row represents a transaction with features such as transaction amount, category, cardholder demographics, and merchant information.

### **Problem Definition**
The central issue with this dataset is **extreme class imbalance**:
- **Non-fraudulent transactions:** ~99.6%
- **Fraudulent transactions:** ~0.4%

This imbalance causes naive models to perform deceptively well on **accuracy**, since predicting “non-fraud” for all transactions yields a 99.6% accuracy — yet detects *zero* frauds.

### **Objective**
Accuracy is a **useless metric** in this scenario.  
Our goal is to **maximize the F1-Score**, which balances **precision** (avoiding false frauds) and **recall** (catching real frauds).

We approach this through a **progressive model gauntlet**, moving from simple linear models to complex ensemble methods — testing both **model architecture** and **resampling strategies**.

### **Methodology Overview**
We evaluate a sequence of models, increasing in complexity and flexibility:
1. **Linear Models:** Logistic Regression, Linear SVM  
2. **Non-linear Models:** KNN, Decision Tree  
3. **Ensemble Models:** Random Forest, XGBoost  

Each is trained on multiple resampled versions of the dataset to test **how well resampling mitigates imbalance**.

---

## **Page 2: Methodology – The Preprocessing Pipeline**

### **Reference**
All preprocessing steps were implemented in **`preprocessing.py`**.

### **Feature Engineering**
To improve signal quality, several meaningful features were engineered:
- **`age`** – Derived from the difference between transaction year and cardholder birth year.
- **`hour`** – Extracted from transaction timestamp to capture time-of-day behavior.
- **`weekday`** – Encodes transaction patterns over the week (e.g., weekday vs weekend behavior).
- **`distance_from_home`** – Geographic feature approximating how far the transaction occurred from the cardholder’s residence.

These features improved interpretability and gave models behavioral context, which is especially critical in fraud detection.

### **Pipeline Design**
Two distinct preprocessing pipelines were designed to match model families:

#### **1. `mode="linear"` (for LogisticRegression, LinearSVC)**
- Used **StandardScaler** for numeric features.
- Used **OneHotEncoder** for categorical features.
- Goal: Normalize input space for models sensitive to feature scale.

#### **2. `mode="tree"` (for DTC, RF, XGB)**
- Used **OrdinalEncoder** for categorical variables.
- No feature scaling (tree-based models are scale-invariant).

All pipeline outputs were saved to CSVs for reproducibility:
- `rf_results.csv`
- `logreg_results.csv`
- `svm_results.csv`, etc.

### **Resampling Strategies**
We experimented with both **upsampling** (minority class duplication) and **downsampling** (majority class reduction):

| Strategy | Description | Reference Dataset |
|-----------|--------------|--------------------|
| **Upsampling (`df_up`)** | Synthetic duplication of minority samples | `*_up.csv` |
| **Downsampling (`df_down`)** | Random majority reduction | `*_down.csv` |

Each strategy was tested under multiple **ratios**:  
`None`, `0.05`, `0.1`, `0.25`, `0.5`, `0.75`, `1.0`  

---

## **Page 3: Results – The “Trash Tier” & Linear Baseline**

### **The “Trash Tier”**
Models like **KNN** and **Naive Bayes** performed disastrously:

| Model | F1 Score |
|--------|-----------|
| KNN | ~0.02 |
| Naive Bayes | ~0.03 |

**KNN** fails due to the **curse of dimensionality** — with 25 features, distance metrics become meaningless and neighborhood density is diluted.  
**Naive Bayes** collapses because it assumes **feature independence**, which is blatantly false in transactional data (e.g., “merchant” and “category” are correlated).

> 📉 These models serve as *control failures*, demonstrating the need for both resampling and stronger learners.

### **The Linear Baseline**
**Logistic Regression** and **LinearSVC** defined the project’s baseline F1 performance.

| Model | F1 Score |
|--------|-----------|
| Logistic Regression | ~0.36–0.38 |
| Linear SVM | ~0.37 |

Both benefited slightly from **upsampling**, especially near a 1:1 balance ratio, but plateaued quickly — confirming that **linear decision boundaries** cannot capture the complex fraud patterns.

### **Graph**
**Bar Chart:** *F1 Score vs. Model (KNN, NB, LR, SVM)*  
> (Insert chart from your results visualization script)

---

## **Page 4: Results – The Tree & Ensemble Gauntlet**

### **Overfitting Lesson: Decision Tree Classifier**
The **DecisionTreeClassifier (DTC)** achieved an artificially high training score but poor generalization:

| Metric | Train F1 | Test F1 |
|---------|-----------|----------|
| DTC | ~0.98 | ~0.42 |

A single tree **memorizes patterns** instead of learning them — classic overfitting. However, it established the foundation for **ensemble learners** to come.

### **Random Forest: The First Real Boost**
Random Forest dramatically improved performance:

| Model | F1 Score |
|--------|-----------|
| Random Forest | ~0.58–0.62 |

Key insights:
- Random subspace sampling mitigated overfitting.
- Model robustness increased across resample ratios.
- Upsampling (ratio = 1.0) gave the most balanced recall vs precision.

### **Extreme Gradient Boosting (XGBoost): The Champion**
XGBoost consistently delivered **the best F1 scores**, typically **>0.70**, outperforming every prior model.

| Model | F1 Score |
|--------|-----------|
| XGBoost | ~0.72–0.75 |

Reasons:
- Gradient boosting adapts to misclassified points.
- Internal handling of imbalance through parameter `scale_pos_weight`.
- Feature importance aligned with intuition — e.g., *amount*, *age*, *distance_from_home*.

---

## **Page 5: Analysis & Discussion**

### **Resampling Observations**
- **Upsampling** generally outperformed **downsampling**, especially for tree-based models.
- **Moderate ratios** (0.25–0.5) provided best tradeoff — extreme upsampling caused redundancy.
- Linear models saturated early; ensembles scaled gracefully.

### **Feature Importance (from XGB)**
Top contributing features:
1. **`amount`**
2. **`distance_from_home`**
3. **`age`**
4. **`hour`**
5. **`category`**

Behavioral and temporal patterns (hour, weekday) showed strong predictive weight, validating the **feature engineering phase**.

### **Error Analysis**
- False negatives were often **low-value, foreign transactions**, implying data sparsity.
- False positives mostly occurred on **high-value domestic transactions** — possibly legitimate anomalies.

---

## **Page 6: Conclusion & Future Work**

### **Conclusion**
This project demonstrated the path **from baseline to boosting** in the fight against class imbalance:
- **Linear models** provided structure but limited predictive power.
- **Tree-based ensembles**, especially **XGBoost**, captured complex, non-linear fraud behaviors.
- **Resampling** proved essential to balancing recall and precision.

| Stage | Best Model | F1 Score |
|--------|-------------|-----------|
| Baseline | Logistic Regression | ~0.37 |
| Tree | Decision Tree | ~0.42 |
| Ensemble | Random Forest | ~0.60 |
| Boosted | XGBoost | ~0.75 |

### **Future Work**
- Integrate **SMOTE** or **ADASYN** synthetic sampling.
- Incorporate **temporal leakage control** for future transactions.
- Explore **stacking ensembles** (e.g., combining RF + XGB).
- Optimize hyperparameters using **Bayesian Optimization**.

### **Final Takeaway**
The journey from naive models to gradient boosting proves one lesson:  
> “Model complexity is meaningless without data balance.  
But once balanced, boosting makes the imbalance irrelevant.”

---

**References**
- scikit-learn documentation (https://scikit-learn.org)
- XGBoost paper: Chen & Guestrin, *“XGBoost: A Scalable Tree Boosting System”*, KDD 2016.
- Dataset: *fraudTrain.csv* (Kaggle Fraud Detection Challenge)
