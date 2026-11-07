import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# PAGE CONFIG
# --------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

st.title("Breast Cancer Classification Dashboard")

st.write(
    "Compare the performance of trained ML classification models."
)

# --------------------------
# MODEL FILES
# --------------------------

model_files = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "SVM": "model/svm.pkl"
}

# --------------------------
# FILE UPLOAD
# --------------------------

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file:

    test_df = pd.read_csv(uploaded_file)

    y_true = test_df["diagnosis"]

    X_test = test_df.drop(
        columns=["diagnosis"]
    )

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    selected_model = st.selectbox(
        "Select Model",
        list(model_files.keys())
    )

    model = joblib.load(
        model_files[selected_model]
    )

    y_pred = model.predict(
        X_test_scaled
    )

    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    st.subheader("Evaluation Metrics")

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC",
            "AUC"
        ],
        "Value": [
            accuracy,
            precision,
            recall,
            f1,
            mcc,
            auc
        ]
    })

    st.dataframe(
        metrics_df.style.format(
            {"Value": "{:.4f}"}
        )
    )

    # --------------------------
    # CONFUSION MATRIX
    # --------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # --------------------------
    # ROC CURVE
    # --------------------------

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    fig2, ax2 = plt.subplots()

    ax2.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.4f}"
    )

    ax2.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax2.set_xlabel(
        "False Positive Rate"
    )

    ax2.set_ylabel(
        "True Positive Rate"
    )

    ax2.legend()

    st.pyplot(fig2)