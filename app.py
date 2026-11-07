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

### Models Included

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)

### Evaluation Metrics

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC

The application loads saved models and evaluates uploaded test data without retraining.
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
1 = Malignant
0 = Benign

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

st.header("📊 Overall Model Performance Comparison")

results_df = pd.read_csv("results.csv")

st.dataframe(
    results_df.round(4),
    use_container_width=True
)

winner_row = results_df.loc[
    results_df["Accuracy"].idxmax()
]

winner_model = winner_row["Model"]

st.success(
    f"""
🏆 Best Performing Model: {winner_model}

Accuracy: {winner_row['Accuracy']:.4f}

AUC: {winner_row['AUC']:.4f}
"""
)

# =====================================================
# UPLOAD TEST DATA
# =====================================================

st.header("📂 Evaluate Test Dataset")

st.info("""
Upload a CSV file containing the same 30 features and
the target column `diagnosis`.

Recommended file:
data/test_data.csv
""")

uploaded_file = st.file_uploader(
    "Upload Test Dataset",
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
    # DATASET INFORMATION
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
    # AVAILABLE MODELS
    # =================================================

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

    # =================================================
    # RE-EVALUATE ALL MODELS ON UPLOADED DATA
    # =================================================

    results_uploaded = []

    for model_name, model_path in model_files.items():

        model = joblib.load(model_path)

        y_pred = model.predict(
            X_test_scaled
        )

        y_prob = model.predict_proba(
            X_test_scaled
        )[:, 1]

        results_uploaded.append([
            model_name,
            accuracy_score(
                y_true,
                y_pred
            ),
            roc_auc_score(
                y_true,
                y_prob
            ),
            precision_score(
                y_true,
                y_pred
            ),
            recall_score(
                y_true,
                y_pred
            ),
            f1_score(
                y_true,
                y_pred
            ),
            matthews_corrcoef(
                y_true,
                y_pred
            )
        ])

    uploaded_results_df = pd.DataFrame(
        results_uploaded,
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

    uploaded_results_df = uploaded_results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    st.header("📈 Results on Uploaded Test Data")

    st.dataframe(
        uploaded_results_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}"
        }),
        use_container_width=True
    )

    # =================================================
    # MODEL ANALYSIS
    # =================================================

    st.header("⚙️ Detailed Model Analysis")

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

    st.subheader(
        f"📈 {selected_model} Metrics"
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    m2.metric(
        "Precision",
        f"{precision:.2%}"
    )

    m3.metric(
        "Recall",
        f"{recall:.2%}"
    )

    m4, m5, m6 = st.columns(3)

    m4.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

    m5.metric(
        "MCC",
        f"{mcc:.4f}"
    )

    m6.metric(
        "AUC",
        f"{auc:.4f}"
    )

    # =================================================
    # PREDICTION RESULTS
    # =================================================

    st.subheader("🔍 Prediction Results")

    prediction_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })

    st.dataframe(
        prediction_df,
        use_container_width=True
    )

    csv = prediction_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name=f"{selected_model}_predictions.csv",
        mime="text/csv"
    )

    # =================================================
    # VISUALIZATIONS
    # =================================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    col1, col2 = st.columns(2)

    # ----------------------------
    # CONFUSION MATRIX
    # ----------------------------

    with col1:

        st.subheader("Confusion Matrix")

        fig1, ax1 = plt.subplots(
            figsize=(5, 4)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax1
        )

        ax1.set_xlabel(
            "Predicted"
        )

        ax1.set_ylabel(
            "Actual"
        )

        st.pyplot(fig1)

    # ----------------------------
    # ROC CURVE
    # ----------------------------

    with col2:

        st.subheader("ROC Curve")

        fig2, ax2 = plt.subplots(
            figsize=(5, 4)
        )

        ax2.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"AUC = {auc:.4f}"
        )

        ax2.plot(
            [0, 1],
            [0, 1],
            "--",
            color="red"
        )

        ax2.set_xlabel(
            "False Positive Rate"
        )

        ax2.set_ylabel(
            "True Positive Rate"
        )

        ax2.set_title(
            selected_model
        )

        ax2.legend()

        st.pyplot(fig2)

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
