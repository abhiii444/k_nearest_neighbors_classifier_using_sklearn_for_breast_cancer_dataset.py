# k_nearest_neighbors_classifier_using_sklearn_for_breast_cancer_dataset.py
This project demonstrates the implementation of a K-Nearest Neighbors (KNN) Classifier using Scikit-learn. It covers the complete machine learning workflow, including data loading, exploratory data analysis , preprocessing, feature scaling, model training with different values of k model evaluation, and saving the trained model using a pipeline.
# 🤖 K-Nearest Neighbors (KNN) Classifier using Scikit-Learn

This project demonstrates an **end-to-end implementation of the K-Nearest Neighbors (KNN) algorithm** using Python and Scikit-learn.  
It includes data preparation, exploratory data analysis, model training, evaluation, and model saving using a pipeline.

---

## 📌 Project Overview

KNN is a **distance-based supervised learning algorithm** that classifies data points based on the majority class of their nearest neighbors.  
This project walks through the complete ML workflow step by step, making it ideal for beginners.

---

## 📂 Project File

k_nearest_neighbors_classifier_using_sklearn_for_breast_cancer_dataset.py


---

## 🔍 Steps Covered in the Project

### 1️⃣ Data Loading
- Dataset loaded using `sklearn.datasets`
- Converted into a Pandas DataFrame
- Saved as a CSV file for reuse

### 2️⃣ Exploratory Data Analysis (EDA)
- Dataset shape and structure
- Summary statistics
- Class distribution
- Missing value check
- Visualizations:
  - Histograms
  - Pair plots
  - Correlation heatmap

### 3️⃣ Data Preprocessing
- Feature and target separation
- Train-test split
- Feature scaling using `StandardScaler`
  > (Important for distance-based models like KNN)

### 4️⃣ Model Training
- Trained KNN classifier with multiple values of **k**
- Selected the best **k** based on accuracy

### 5️⃣ Model Evaluation
- Accuracy score
- Confusion matrix
- Classification report
- Heatmap visualization of confusion matrix

### 6️⃣ Model Saving
- Created a pipeline (Scaler + KNN)
- Saved trained model using `joblib`

### 7️⃣ Model Inference
- Loaded the saved model
- Predicted class for new sample data

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## 🚀 How to Run the Project

1. Install required libraries:
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn joblib
2. Run the script:
    ```bash
    python k_nearest_neighbors_classifier_using_sklearn_for_breast_cancer_dataset.py

---
## 📊 Model Output
- Best value of k
- Accuracy on test data
- Confusion matrix
- Classification report
- Saved model file:
    ```bash
    knn_iris_model.pkl

---

## Concepts Learned
- KNN algorithm
- Feature scaling importance
- Train-test split
- Hyperparameter tuning
- Model evaluation metrics
- ML pipelines
- Model persistence

---
