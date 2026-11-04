import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# 1. Download UCI Adult Census Dataset (14 features > 12 minimum; 6000 rows > 500 minimum)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

print("1/4: Downloading dataset from UCI...")
df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
df = df.dropna().reset_index(drop=True)

# Sample 6,000 instances for fast execution and compact model artifacts (<25MB)
df = df.sample(n=6000, random_state=42).reset_index(drop=True)

# Encode binary target: 0 for <=50K, 1 for >50K
df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

X = df.drop(columns=["income"])
y = df["income"]

# 2. Train-Test Split (80% Train, 20% Test)
print("2/4: Splitting train/test partitions and saving test_data.csv...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Save test dataset (features + target) for Streamlit upload testing
test_df = X_test.copy()
test_df["income"] = y_test.values
test_df.to_csv("test_data.csv", index=False)

# 3. Setup Preprocessor Pipeline
numeric_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
categorical_features = [col for col in X.columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

os.makedirs("model", exist_ok=True)
joblib.dump(preprocessor, "model/preprocessor.joblib")

# 4. Train the 5 Required Classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
}

print("3/4: Training 5 models and exporting joblib artifacts...")
results = []
for name, clf in models.items():
    clf.fit(X_train_proc, y_train)
    y_pred = clf.predict(X_test_proc)
    y_proba = clf.predict_proba(X_test_proc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })

    slug = name.lower().replace(" ", "_")
    joblib.dump(clf, f"model/{slug}.joblib")
    print(f" -> Trained & exported: {name}")

# 5. Output Summary Table for README
print("4/4: Execution complete!\n")
results_df = pd.DataFrame(results)
print("=== COPY THIS METRICS TABLE FOR YOUR README ===")
print(results_df.to_string(index=False))
