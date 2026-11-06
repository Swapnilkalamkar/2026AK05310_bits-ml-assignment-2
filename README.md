# Machine Learning Assignment - 2: Classification Models & Streamlit Deployment

## a. Problem Statement
The objective of this assignment is to develop, train, evaluate, and deploy five distinct supervised machine learning classification algorithms on a standardized tabular dataset. The models predict whether an individual earns more than $50,000 per year based on demographic and employment attributes. The trained artifacts are deployed as an interactive Streamlit web dashboard.

## b. Dataset Description
* **Dataset Name**: UCI Adult / Census Income Dataset
* **Source**: UCI Machine Learning Repository
* **Total Instances**: 48,842 (Sampled 6,000 clean instances for modeling)
* **Total Features**: 14 input attributes (6 numerical, 8 categorical)
  * Numerical: `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
  * Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
* **Target Feature**: `income` (Binary: `<=50K` [0] vs `>50K` [1])

## c. GitHub Repository Link
* **Repository**: https://github.com/Swapnilkalamkar/2026AK05310_bits-ml-assignment-2
* **Live Streamlit App**: https://2026ak05310bits-ml-assignment-2.streamlit.app

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8633 | 0.9088 | 0.7773 | 0.6117 | 0.6846 | 0.6060 |
| Decision Tree | 0.8642 | 0.9013 | 0.7783 | 0.6151 | 0.6871 | 0.6087 |
| KNN | 0.8525 | 0.8808 | 0.7244 | 0.6323 | 0.6752 | 0.5826 |
| Naive Bayes | 0.3467 | 0.5826 | 0.2677 | 0.9759 | 0.4201 | 0.1623 |
| Support Vector Machine | 0.8683 | 0.9006 | 0.7854 | 0.6289 | 0.6985 | 0.6218 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Provides a solid linear baseline with balanced precision (0.7773) and high AUC (0.9088). Fast inference and well-calibrated probabilities. |
| Decision Tree | Captures non-linear feature splits well with controlled depth (`max_depth=8`), yielding a strong F1 score (0.6871) without severe overfitting. |
| KNN | Sensitive to high-dimensional one-hot encoded spaces; achieves a reasonable accuracy of 0.8525, but exhibits a lower MCC score (0.5826) compared to linear/kernel baselines. |
| Naive Bayes | Demonstrates high Recall (0.9759) but low Precision (0.2677) and Accuracy (0.3467) due to strong feature conditional independence assumptions violated by correlated census features. |
| Support Vector Machine | Best overall performer across Accuracy (0.8683), Precision (0.7854), AUC (0.9306), F1 (0.6985), and MCC (0.6218) using an RBF kernel. |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)** achieved the highest performance across all primary metrics on the test partition. |
