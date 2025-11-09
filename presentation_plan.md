# Slide 1: Title
## Detecting Credit Card Fraud: A Comparative Analysis of Machine Learning Models
**Your Name**  
**Your Class / Course**

---

# Slide 2: The Problem – Extreme Imbalance
### “A Needle in a Haystack”
- **Dataset:** ~1.8 million transactions  
- **Class Distribution:**  
  - 99.6% Legitimate  
  - 0.4% Fraud  
- **Visualization:** Pie Chart → Legitimate (huge blue slice), Fraud (tiny red slice)  
- **Key Point:**  
  Accuracy ≈ useless. Predicting “no fraud” gives 99.6% accuracy but catches nothing.  
  **Goal:** Maximize **F1-Score**, not accuracy.

---

# Slide 3: The Data & Preprocessing Pipeline
### Overview Diagram
**Flow:**  
Raw Data → `preprocessing.py` →  
Feature Engineering → Two Pipelines → Resampling  

**Feature Engineering:**  
- `age` (from DOB)  
- `hour` (from timestamp)  
- `weekday`  
- `distance_from_home`  

**Pipelines:**  
- **Linear Mode:** StandardScaler + OneHotEncoder  
- **Tree Mode:** OrdinalEncoder (no scaling)  

**Resampling Strategies:**  
- **Upsampling (df_up)** – duplicate fraud samples  
- **Downsampling (df_down)** – reduce legit samples  

---

# Slide 4: The “Trash Tier” (F1 < 0.05)
### Models That Completely Failed
| Model | F1 Score |
|--------|-----------|
| KNN | ~0.02 |
| Naive Bayes | ~0.03 |

- **Visual:** Big red “FAIL” stamp over both bars.  
- **Why They Failed:**  
  - **KNN:** Victim of the *Curse of Dimensionality* – 25 features make distance meaningless.  
  - **Naive Bayes:** Assumes all features are independent (they aren’t).  

📉 *Takeaway:* These are “proof of failure” baselines to show what *not* to use.

---

# Slide 5: The Linear Baseline (F1 ≈ 0.37)
| Model | F1 Score |
|--------|-----------|
| Logistic Regression | ~0.36 |
| Linear SVM | ~0.37 |

**Visual:** Bar chart labeled “Score to Beat.”  
**Interpretation:**
- These models provide a clean, interpretable baseline.
- Limited by linear separability — fraud patterns are highly non-linear.  
- Scaling helps, but **boundaries remain too simple**.  

📈 *Key Point:* Linear models can’t capture the complex fraud behavior.

---

# Slide 6: The “Aha!” Moment – Overfitting
| Model | F1 Score |
|--------|-----------|
| DecisionTree (Full Depth) | 0.46 |
| DecisionTree (Depth=10) | 0.56 |

**Visual:** Two trees side-by-side; one tangled and massive (overfit), one pruned (clean).  
**Insight:**  
- Unrestricted trees memorize the training set.  
- Pruning limits depth → forces generalization.  
- F1 jumped by 10 points after pruning.  

🌳 *Lesson:* Simplicity beats memorization.

---

# Slide 7: The Ensemble Fix – RF & XGBoost
| Model | F1 Score |
|--------|-----------|
| Logistic Regression | 0.37 |
| Decision Tree (Depth=10) | 0.56 |
| Random Forest | 0.83 |
| XGBoost | 0.85 |

**Visual:** Line chart showing the climb from 0.37 → 0.85.  
**Key Insight:**  
Ensembles (RF, XGB) **combine many weak trees** to form a powerful learner.  
They reduce variance and fix overfitting.  

🔥 *Turning Point:* The models finally understand fraud patterns instead of memorizing noise.

---

# Slide 8: The Winning Strategy – Ratio
### The “Boy Who Cried Wolf” Graph
**Graph:** F1 Score vs. Resampling Ratio  
- X-axis: ratio (0.05 → 1.0)  
- Y-axis: F1 Score  
- Curve peaks near **0.1**, then collapses.

**Key Insight:**  
- At **ratio=1.0**, model overfits to fake data.  
- At **ratio=0.1**, best balance between recall and precision.  

⚠️ *Conclusion:* “All-in upsampling” is a **trash strategy**. Controlled balance wins.

---

# Slide 9: The Winning Strategy – Sampling
| Model | Dataset | F1 Score |
|--------|----------|-----------|
| XGBoost | df_down | 0.63 |
| XGBoost | df_up | 0.85 |

**Visual:** Simple table or two-bar comparison.  
**Insight:**  
- Upsampling gives model more minority examples to learn from.  
- Downsampling throws away valuable data.  
- **XGB + df_up** is the winning combo.  

🚀 *Takeaway:* Complex models thrive on more (balanced) data.

---

# Slide 10: Conclusion & Demo
### Final Champion
**Model:** XGBoost  
**Sampling:** `df_up`  
**Ratio:** 0.1  
**Final F1 Score:** **0.851**

**Visual:** Screenshot of Streamlit demo (fraud detection dashboard).  

🧠 **Final Thoughts:**  
- Resampling and feature engineering matter as much as model choice.  
- Boosted ensembles dominate when tuned properly.  
- Fraud detection is about *balance*, not brute force.

💬 **“Questions?”**
