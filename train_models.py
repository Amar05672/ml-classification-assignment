import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load dataset
df = pd.read_csv("Breast_Cancer_Wisconsin.csv")

# Remove unwanted columns
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Encode target
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# Split features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save test data for Streamlit
test_df = X_test.copy()
test_df["diagnosis"] = y_test.values
test_df.to_csv("data/test_data.csv", index=False)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

# Models
models = {
    "Logistic Regression":
        LogisticRegression(max_iter=2000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC(probability=True, random_state=42)
}

results = []

for name, model in models.items():

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    y_prob = model.predict_proba(X_test_scaled)[:,1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    mcc = matthews_corrcoef(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        mcc,
        auc
    ])

# Save models
joblib.dump(
    models["Logistic Regression"],
    "model/logistic.pkl"
)

joblib.dump(
    models["Decision Tree"],
    "model/decision_tree.pkl"
)

joblib.dump(
    models["KNN"],
    "model/knn.pkl"
)

joblib.dump(
    models["Naive Bayes"],
    "model/naive_bayes.pkl"
)

joblib.dump(
    models["SVM"],
    "model/svm.pkl"
)

# Save results
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "AUC"
    ]
)

results_df.to_csv(
    "results.csv",
    index=False
)

print("\nTraining Complete!")
print(results_df)