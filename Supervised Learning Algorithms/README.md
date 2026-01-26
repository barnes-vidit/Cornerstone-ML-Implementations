
In Supervised Learning, the model learns from **labeled data**. It's like a student learning from a teacher who provides the questions (Features, $X$) and the correct answers (Target, $y$).

**Goal:** Learn a mapping function $y = f(X)$ to predict outputs for new, unseen data.

---

## ⚖️ Core Concepts (The Interview Stuff)

### 1. Bias-Variance Tradeoff (The "Goldilocks" Problem)
*   **High Bias (Underfitting):** The model is too simple. It misses the patterns.
    *   *Example:* Trying to fit a curve with a straight line.
    *   *Solution:* Use a more complex model (e.g., switch from Logistic Regression to Random Forest).
*   **High Variance (Overfitting):** The model is too complex. It memorizes the noise in the training data.
    *   *Example:* Connecting every single dot in a scatter plot.
    *   *Solution:* Get more data, simplify the model (Pruning), or use Regularization (L1/L2).

### 2. Class Imbalance
When one class dominates the other (e.g., 99% benign, 1% fraud).
*   **Problem:** The model achieves 99% accuracy by simply guessing "Benign" every time. It learns nothing.
*   **Solutions:**
    *   **Resampling:** Oversample the minority class (SMOTE) or Undersample the majority.
    *   **Class Weights:** Tell the model "This class is 10x more important" (available in most sklearn models).
    *   **Metric:** Stop looking at Accuracy. Look at **Recall** or F1-Score.

### 3. Data Splitting
*   **Train Set:** Books the model studies from.
*   **Test Set:** The final exam. (NEVER touch this during training).
*   **Validation Set:** Practice exams to tune the model's settings (Hyperparameters).

---

## 📊 Evaluation (How do we know it works?)

| Metric | Definition | When to use? |
| :--- | :--- | :--- |
| **Accuracy** | Correct Guesses / Total | Only when classes are balanced. |
| **Precision** | True Positives / (True Positives + False Positives) | When False Positives are bad (e.g., Spam Filter - don't block real email). |
| **Recall** | True Positives / (True Positives + False Negatives) | When False Negatives are bad (e.g., Cancer Diagnosis - don't miss a case). |
| **F1-Score** | Harmonic Mean of Precision & Recall | Good balance. Best for imbalanced data. |
| **ROC-AUC** | Area Under Curve | To see how well the model separates classes at different thresholds. |

---

## 🛠️ The Cheat Sheet: Which Algorithm to Use?

| Algorithm | Pros | Cons | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Simple, Fast, Explainable (Odds). | Linear boundaries only. Can't handle complex patterns. | Binary Classification Baselines. |
| **KNN (K-Nearest Neighbors)** | No training phase. Simple logic. | Slow on large data. Sensitive to scale. | Small datasets, quick groupings. |
| **SVM (Support Vector Machine)** | Great for high dimensions. Effective with Kernels. | Slow on large data. Hard to tune. | Image recognition, complex small datasets. |
| **Naive Bayes** | Extremely fast. Works well with high dimensions. | Assumes feature independence (rarely true). | **Text/Spam Classification**. |
| **Decision Tree** | "White Box" (Visualizable). No scaling needed. | Prone to Overfitting. Unstable. | When you need to explain "Why". |
| **Random Forest** | Robust. Hard to overfit. Handles non-linear data well. | Slower to predict. "Black Box". | **General Purpose Winner**. |
| **Gradient Boosting (XGB/LGBM/Cat)** | **State-of-the-Art Accuracy.** Wins competitions. | Harder to tune. Prone to overfitting on noise. | **Tabular Data Competitions.** |

---

## 📂 The Files
This folder contains:
1.  `Regression.ipynb`: Predicting continuous values.
2.  `Logistic Regression.ipynb`: The baseline classifier.
3.  `KNN.ipynb`: Distance-based classification.
4.  `SVM.ipynb`: Finding the best margin.
5.  `Naive Bayes.ipynb`: Probabilistic classification.
6.  `Decision Tree.ipynb`: Rule-based logic.
7.  `Random Forest.ipynb`: Bagging (Ensemble).
8.  `Gradient Boosting.ipynb`: Boosting (Ensemble).
9.  `Advanced Boosting.ipynb`: XGBoost vs LightGBM vs CatBoost.
