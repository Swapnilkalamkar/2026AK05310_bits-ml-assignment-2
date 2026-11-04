import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Adult Census Income Classifier", layout="wide")

st.title("Adult Census Income - Model Performance Evaluation")
st.write("Upload the test CSV data to evaluate pre-trained models without retraining.")

@st.cache_resource
def load_preprocessor():
    return joblib.load("model/preprocessor.joblib")

@st.cache_resource
def load_model(model_name):
    model_mapping = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree": "model/decision_tree.joblib",
        "KNN": "model/knn.joblib",
        "Naive Bayes": "model/naive_bayes.joblib",
        "Support Vector Machine": "model/support_vector_machine.joblib"
    }
    return joblib.load(model_mapping[model_name])

# Sidebar Controls
st.sidebar.header("Controls & Setup")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (test_data.csv)", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Select Classifier Model",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine")
)

target_col = st.sidebar.text_input("Target Column Name", value="income")

if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Test Data Preview")
        st.write(f"Shape: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
        st.dataframe(df_test.head(5))

        if target_col not in df_test.columns:
            st.error(f"Target column '{target_col}' not found in uploaded CSV.")
        else:
            y_test = df_test[target_col].astype(int).values
            X_test_raw = df_test.drop(columns=[target_col])

            preprocessor = load_preprocessor()
            model = load_model(model_choice)

            # Transform raw test features using saved preprocessor
            X_test_proc = preprocessor.transform(X_test_raw)

            # Predictions
            y_pred = model.predict(X_test_proc)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_proc)[:, 1]
            else:
                y_proba = model.decision_function(X_test_proc)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.markdown(f"### Evaluation Metrics: **{model_choice}**")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("ROC-AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1-Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.markdown("---")
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                ax.set_xticklabels(["<=50K (0)", ">50K (1)"])
                ax.set_yticklabels(["<=50K (0)", ">50K (1)"])
                st.pyplot(fig)

            with c2:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format(precision=4))

    except Exception as e:
        st.error(f"Error executing model inference: {str(e)}")
else:
    st.info("Please upload test_data.csv via the sidebar to view model evaluation.")
