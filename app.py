import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🩺 Breast Cancer Classification Dashboard")

st.markdown("""
### Project Overview

This project compares multiple machine learning classification algorithms
for predicting whether a breast tumor is:

- Malignant (1)
- Benign (0)

using the Breast Cancer Wisconsin Dataset.

The application loads previously trained models and evaluates
them on uploaded test data without retraining.
""")

# =====================================================
# DATASET SECTION
# =====================================================

st.header("📋 Dataset Information")

c1, c2, c3 = st.columns(3)

c1.metric("Total Samples", "569")
c2.metric("Input Features", "30")
c3.metric("Classes", "2")

st.info("""
Dataset: Breast Cancer Wisconsin Dataset

Target Variable: diagnosis

Classes:
- 1 = Malignant
- 0 = Benign

Preprocessing:
- Removed id column
- Removed Unnamed: 32 column
- Applied StandardScaler
""")

# =====================================================
# MODELS
# =====================================================

st.header("🤖 Models Evaluated")

st.markdown("""
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
""")

# =====================================================
# METRICS
# =====================================================

st.header("📏 Evaluation Metrics")

st.markdown("""
The following metrics are used:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
""")

# =====================================================
# PROJECT WORKFLOW
# =====================================================

st.header("⚙️ Project Workflow")

st.info("""
Dataset
➡ Preprocessing
➡ Train/Test Split
➡ Model Training
➡ Save Models (.pkl)
➡ Streamlit Deployment
➡ Model Evaluation
""")

# =====================================================
# UPLOAD DATA
# =====================================================

st.header("📂 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    if "diagnosis" not in test_df.columns:

        st.error(
            "The uploaded file must contain the diagnosis column."
        )
        st.stop()

    # =================================================
    # DATA SUMMARY
    # =================================================

    y_true = test_df["diagnosis"]

    X_test = test_df.drop(
        columns=["diagnosis"]
    )

    st.subheader("📊 Uploaded Dataset Summary")

    d1, d2, d3 = st.columns(3)

    d1.metric("Rows", len(test_df))
    d2.metric("Features", X_test.shape[1])
    d3.metric("Target", "Diagnosis")

    # =================================================
    # LOAD SCALER
    # =================================================

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # =================================================
    # LOAD MODELS
    # =================================================

    model_paths = {
        "Logistic Regression": "model/logistic.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "SVM": "model/svm.pkl"
    }

    results = []

    roc_data = {}

    confusion_matrices = {}

    predictions_store = {}

    # =================================================
    # EVALUATE ALL MODELS
    # =================================================

    for model_name, model_path in model_paths.items():

        model = joblib.load(model_path)

        y_pred = model.predict(X_test_scaled)

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

        results.append([
            model_name,
            accuracy,
            auc,
            precision,
            recall,
            f1,
            mcc
        ])

        roc_data[model_name] = roc_curve(
            y_true,
            y_prob
        )

        confusion_matrices[model_name] = (
            confusion_matrix(
                y_true,
                y_pred
            )
        )

        predictions_store[model_name] = y_pred

    # =================================================
    # RESULTS TABLE
    # =================================================

    st.header("📈 Model Performance Comparison")

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC"
        ]
    )

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    st.dataframe(
        results_df.style.format({
            "Accuracy":"{:.4f}",
            "AUC":"{:.4f}",
            "Precision":"{:.4f}",
            "Recall":"{:.4f}",
            "F1":"{:.4f}",
            "MCC":"{:.4f}"
        }),
        use_container_width=True
    )

    # =================================================
    # WINNER
    # =================================================

    winner_row = results_df.iloc[0]

    st.success(
        f"""
🏆 Best Performing Model: {winner_row['Model']}

Accuracy: {winner_row['Accuracy']:.4f}

AUC: {winner_row['AUC']:.4f}
"""
    )

    # =================================================
    # ROC COMPARISON
    # =================================================

    st.header("📉 ROC Curve Comparison")

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for model_name in roc_data:

        fpr, tpr, _ = roc_data[model_name]

        auc_value = results_df.loc[
            results_df["Model"] == model_name,
            "AUC"
        ].values[0]

        ax.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC={auc_value:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="black"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve Comparison"
    )

    ax.legend()

    st.pyplot(fig)

    # =================================================
    # CONFUSION MATRICES
    # =================================================

    st.header("📊 Confusion Matrices")

    cols = st.columns(2)

    for i, model_name in enumerate(confusion_matrices):

        cm = confusion_matrices[model_name]

        with cols[i % 2]:

            st.subheader(model_name)

            fig_cm, ax_cm = plt.subplots()

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax_cm
            )

            ax_cm.set_xlabel(
                "Predicted"
            )

            ax_cm.set_ylabel(
                "Actual"
            )

            st.pyplot(fig_cm)

    # =================================================
    # PREDICTION RESULTS
    # =================================================

    st.header("🔍 Prediction Results")

    selected_prediction_model = st.selectbox(
        "Select Model",
        list(model_paths.keys())
    )

    prediction_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted":
            predictions_store[
                selected_prediction_model
            ]
    })

    st.dataframe(
        prediction_df,
        use_container_width=True
    )

    csv = prediction_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download Predictions",
        csv,
        file_name=f"{selected_prediction_model}_predictions.csv",
        mime="text/csv"
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption("""
Breast Cancer Classification Dashboard

Models:
Logistic Regression | Decision Tree | KNN | Naive Bayes | SVM

Evaluation Metrics:
Accuracy | AUC | Precision | Recall | F1 | MCC

Author: Amarnath Krishnan
""")
