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

Breast cancer is one of the most common forms of cancer worldwide.
Early and accurate diagnosis can significantly improve treatment outcomes.

This project evaluates five Machine Learning classification models that
predict whether a breast tumor is:

- **Malignant (Cancerous)**
- **Benign (Non-Cancerous)**

The models were trained offline and saved as `.pkl` files.
This Streamlit application loads the saved models and evaluates
their performance using a test dataset.
""")

# =====================================================
# DATASET INFORMATION
# =====================================================

st.header("📋 Dataset Information")

c1, c2, c3 = st.columns(3)

c1.metric("Total Samples", "569")
c2.metric("Features", "30")
c3.metric("Classes", "2")

st.info("""
**Dataset:** Breast Cancer Wisconsin Dataset

**Target Variable:** diagnosis

**Class Labels**

- 1 = Malignant
- 0 = Benign

**Preprocessing**

- Dropped id column
- Dropped Unnamed: 32 column
- Applied StandardScaler
""")

# =====================================================
# MODELS
# =====================================================

st.header("🤖 Models Evaluated")

st.markdown("""
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)
""")

# =====================================================
# METRICS
# =====================================================

st.header("📏 Evaluation Metrics")

st.markdown("""
The following metrics are used to compare model performance:

- Accuracy
- Area Under Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
""")

# =====================================================
# RESULTS COMPARISON
# =====================================================

st.header("📊 Model Performance Comparison")

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
"""
)

# =====================================================
# MODEL OBSERVATIONS
# =====================================================

st.header("📝 Model Observations")

observations = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine"
    ],
    "Observation": [
        "Produced excellent overall performance with very high AUC and accuracy.",
        "Easy to interpret but achieved slightly lower performance than the top models.",
        "Provided strong classification performance after feature scaling.",
        "Fast and effective model despite its simplifying assumptions.",
        "Achieved the best balance of Accuracy, Precision, F1 Score and MCC."
    ]
})

st.dataframe(
    observations,
    use_container_width=True
)

st.info("""
🏆 **Overall Winner: Support Vector Machine (SVM)**

The SVM model achieved the highest overall performance and was selected as the best classifier for this dataset.
""")

# =====================================================
# WORKFLOW
# =====================================================

st.header("⚙️ Project Workflow")

st.info("""
Dataset
➡ Preprocessing
➡ Train-Test Split
➡ Model Training
➡ Save Trained Models (.pkl)
➡ Streamlit Deployment
➡ Model Evaluation
""")

# =====================================================
# MODEL PATHS
# =====================================================

model_paths = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "SVM": "model/svm.pkl"
}

# =====================================================
# TEST DATA EVALUATION
# =====================================================

st.header("📂 Evaluate Test Dataset")

st.markdown("""
Upload a CSV file containing the same feature columns used during training.

**Recommended File**

`data/test_data.csv`
""")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    if "diagnosis" not in test_df.columns:

        st.error(
            "Uploaded file must contain the 'diagnosis' column."
        )
        st.stop()

    y_true = test_df["diagnosis"]

    X_test = test_df.drop(
        columns=["diagnosis"]
    )

    # ==========================================
    # DATASET INFO
    # ==========================================

    st.subheader("📋 Uploaded Dataset Summary")

    d1, d2, d3 = st.columns(3)

    d1.metric("Rows", len(test_df))
    d2.metric("Features", X_test.shape[1])
    d3.metric("Target", "Diagnosis")

    # ==========================================
    # LOAD SCALER
    # ==========================================

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ==========================================
    # MODEL SELECTION
    # ==========================================

    selected_model = st.selectbox(
        "Select Classification Model",
        list(model_paths.keys())
    )

    model = joblib.load(
        model_paths[selected_model]
    )

    y_pred = model.predict(
        X_test_scaled
    )

    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # ==========================================
    # METRICS
    # ==========================================

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

    st.header("📈 Model Performance")

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

    # ==========================================
    # PREDICTIONS
    # ==========================================

    st.header("🔍 Prediction Results")

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
        file_name="predictions.csv",
        mime="text/csv"
    )

    # ==========================================
    # VISUALIZATIONS
    # ==========================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    col1, col2 = st.columns(2)

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

    with col2:

        st.subheader("ROC Curve")

        fig2, ax2 = plt.subplots(
            figsize=(5, 4)
        )

        ax2.plot(
            fpr,
            tpr,
            label=f"AUC = {auc:.4f}",
            linewidth=2
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

        ax2.legend()

        st.pyplot(fig2)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption("""
Breast Cancer Classification Dashboard

Dataset: Breast Cancer Wisconsin Dataset

Developed using:
- Streamlit
- Scikit-Learn
- Pandas
- Matplotlib
- Seaborn

Author: Amarnath Krishnan
""")
