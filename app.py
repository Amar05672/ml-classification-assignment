import streamlit as st
import pandas as pd
import joblib

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

# =====================================================
# PAGE CONFIGURATION
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
This application evaluates **five pre-trained machine learning classifiers**
for predicting whether a breast tumor is **Malignant (M)** or **Benign (B)**.

Models included:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Project Information")

st.sidebar.info("""
Dataset:
Breast Cancer Wisconsin Dataset

Target Variable:
diagnosis

Classes:
- 1 = Malignant
- 0 = Benign

Models:
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- SVM
""")

# =====================================================
# MODEL COMPARISON
# =====================================================

st.header("📊 Model Performance Comparison")

results_df = pd.read_csv("results.csv")

st.dataframe(
    results_df.round(4),
    use_container_width=True
)

winner = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model"
]

st.success(f"🏆 Best Performing Model: {winner}")

# =====================================================
# UPLOAD TEST DATA
# =====================================================

st.header("📂 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    required_columns = list(test_df.columns)

    if "diagnosis" not in required_columns:

        st.error(
            "The uploaded file must contain the 'diagnosis' column."
        )
        st.stop()

    # =================================================
    # DATASET INFO
    # =================================================

    y_true = test_df["diagnosis"]

    X_test = test_df.drop(
        columns=["diagnosis"]
    )

    st.subheader("📋 Dataset Information")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        len(test_df)
    )

    c2.metric(
        "Features",
        X_test.shape[1]
    )

    c3.metric(
        "Target",
        "Diagnosis"
    )

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
    # MODEL SELECTION
    # =================================================

    st.header("⚙️ Select Model")

    model_files = {
        "Logistic Regression":
            "model/logistic.pkl",

        "Decision Tree":
            "model/decision_tree.pkl",

        "KNN":
            "model/knn.pkl",

        "Naive Bayes":
            "model/naive_bayes.pkl",

        "SVM":
            "model/svm.pkl"
    }

    selected_model = st.selectbox(
        "Choose a machine learning model",
        list(model_files.keys())
    )

    model = joblib.load(
        model_files[selected_model]
    )

    # =================================================
    # PREDICTION
    # =================================================

    y_pred = model.predict(
        X_test_scaled
    )

    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # =================================================
    # METRICS
    # =================================================

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

    st.header(
        f"📈 {selected_model} Performance"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    col2.metric(
        "Precision",
        f"{precision:.2%}"
    )

    col3.metric(
        "Recall",
        f"{recall:.2%}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

    col5.metric(
        "MCC",
        f"{mcc:.4f}"
    )

    col6.metric(
        "AUC",
        f"{auc:.4f}"
    )

    # =================================================
    # PREDICTION TABLE
    # =================================================

    st.header("🔍 Prediction Results")

    prediction_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })

    st.dataframe(
        prediction_df.head(30),
        use_container_width=True
    )

    # =================================================
    # DOWNLOAD BUTTON
    # =================================================

    csv = prediction_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

    # =================================================
    # CONFUSION MATRIX
    # =================================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    # =================================================
    # SIDE-BY-SIDE CHARTS
    # =================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Confusion Matrix")

        fig1, ax1 = plt.subplots(
            figsize=(5, 4)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax1
        )

        ax1.set_xlabel(
            "Predicted"
        )

        ax1.set_ylabel(
            "Actual"
        )

        st.pyplot(fig1)

    with col2:

        st.subheader("ROC Curve")

        fig2, ax2 = plt.subplots(
            figsize=(5, 4)
        )

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

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
"""
Breast Cancer Classification Dashboard using Streamlit and Scikit-Learn.

Author: Amarnath Krishnan
"""
)
