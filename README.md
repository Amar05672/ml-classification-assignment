# Brest Cancer predictor

## a. Problem Statement
Breast cancer is one of the most common forms of cancer affecting individuals worldwide. Early and accurate diagnosis plays a critical role in improving treatment outcomes and reducing mortality rates.

The objective of this project is to develop and evaluate machine learning classification models capable of predicting whether a breast tumor is benign (non-cancerous) or malignant (cancerous) based on tumor characteristics extracted from digitized fine needle aspirate (FNA) images of breast masses.

A Streamlit-based Machine Learning Classification Dashboard was developed to compare the performance of multiple classification algorithms and identify the most suitable model for breast cancer diagnosis.

The application should:

- Load trained machine learning models.
- Evaluate classification performance on test data.
- Compare model performance using multiple evaluation metrics.
- Display visualizations such as Confusion Matrix and ROC Curve.

---

## b. Dataset Description

Dataset Name:
Breast Cancer Wisconsin Dataset

Dataset_Source : UCI

url=https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Objective:
Predict whether a breast tumor is:

- Malignant (M)
- Benign (B)

Dataset Characteristics:

- Number of Samples: 569
- Number of Features: 30
- Target Variable: diagnosis
- Classes:
  - M = Malignant
  - B = Benign

Preprocessing Performed:

- Removed `id` column
- Removed `Unnamed: 32` column
- Converted target labels:
  - M → 1
  - B → 0
- Applied StandardScaler for feature normalization

---

## c. GitHub Repository Link

Repository:

https://github.com/Amar05672/ml-classification-assignment

Streamlit app:

https://ml-classification-assignment-6hwbpp2bpmup7igfbshtzz.streamlit.app/

---

## Project Structure

ML_Classification_Assignment/

├── app.py

├── train_models.py

├── requirements.txt

├── README.md

├── results.csv

├── model/

│   ├── logistic.pkl

│   ├── decision_tree.pkl

│   ├── knn.pkl

│   ├── naive_bayes.pkl

│   ├── svm.pkl

│   └── scaler.pkl

└── data/

    └── test_data.csv

---

## d. Models Used

The following classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

### Performance Comparison

| ML Model Name	| Accuracy| Precision  | Recall | F1 |	MCC	| AUC |
|-----------|--------|---|------|-----|--|---|
|Logistic Regression	|0.965|	0.975|	0.929|	0.951|	0.925|	0.996|
Decision Tree	|0.930	|0.905|	0.905	|0.905|	0.849|	0.925|
KNN	|0.956	|0.974	|0.905	|0.938|	0.906|	0.982|
Naive Bayes	|0.921	|0.923	|0.857	|0.889|	0.829|	0.989|
Support Vector Machine (SVM)	|0.974|	1.000	|0.929|	0.963|	0.944	|0.995|

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|------------------------------------|
| Logistic Regression | Achieved strong predictive performance with excellent precision and recall. Suitable for linearly separable patterns in the dataset. |
| Decision Tree | Easily interpretable model. Performance was slightly affected by overfitting compared to other classifiers. |
| KNN | Produced competitive classification results. Performance depended on neighborhood selection and feature scaling. |
| Naive Bayes | Fast and efficient classifier. Performed reasonably well despite independence assumptions between features. |
| Support Vector Machine | Produced high classification accuracy and strong discrimination capability. Handled feature space effectively after scaling. |

### Overall Winner for the Dataset

Winner Model: Support Vector Machine (SVM)

_____________________

Reason:

The Support Vector Machine achieved:

- Highest Accuracy (97.37%)
- Highest Precision (100%)
- Highest F1 Score (96.30%)
- Highest MCC (0.944)

Although Logistic Regression achieved a slightly higher AUC (99.60% vs 99.47%), SVM demonstrated the best overall balance across all evaluation metrics and was therefore selected as the best-performing model for the Breast Cancer Wisconsin Dataset.


## Running the Application

- Install dependencies:
- pip install -r requirements.txt
- Run Streamlit application:
- streamlit run app.py

## Important Notes:

- The uploaded CSV file must contain the same 30 feature columns used during model training and the target column diagnosis.
- The id column and Unnamed: 32 column are not used during model evaluation.
- The sample test dataset used in this project is provided as data/test_data.csv.
- The application loads only the pre-trained models stored in the model/ directory.
- No model training is performed within the Streamlit application.
- All predictions and performance evaluations are generated using the previously trained and saved classifiers.

Select a model from the dropdown and evaluate its performance using the displayed metrics and visualizations.
