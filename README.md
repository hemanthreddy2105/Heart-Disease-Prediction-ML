# ❤️ Heart Disease Prediction using Machine Learning

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help healthcare professionals make timely decisions and improve patient outcomes.

This project develops an end-to-end Machine Learning model to predict whether a patient is likely to have heart disease based on various clinical attributes. Multiple classification algorithms are trained and compared, and the best-performing model is optimized using Hyperparameter Tuning.

---

## 🎯 Objective

The objective of this project is to:

- Perform Exploratory Data Analysis (EDA)
- Preprocess healthcare data
- Train multiple Machine Learning models
- Compare model performance
- Select the best model
- Optimize the model using GridSearchCV
- Save the trained model for future use

---

## 📂 Dataset

The dataset contains medical information collected from patients.

### Features

| Feature | Description |
|----------|-------------|
| age | Age of the patient |
| sex | Gender |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |
| condition | Target Variable (Heart Disease) |

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- gogole collab Notebook

---

# 📊 Project Workflow

```
Dataset
   │
   ▼
Data Understanding
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Data Preprocessing
   │
   ▼
Train-Test Split
   │
   ▼
Feature Scaling
   │
   ▼
Model Building
   │
   ▼
Model Comparison
   │
   ▼
Best Model Selection
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Feature Importance
   │
   ▼
Model Saving
```

---

# 🤖 Machine Learning Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

---

# 📈 Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

# 🏆 Best Performing Model

After evaluating multiple machine learning algorithms, **Logistic Regression** achieved the best overall performance.

The model was further optimized using **GridSearchCV**, resulting in improved prediction performance.

---

# 📷 Visualizations

The project includes:

- Target Variable Distribution
- Histograms
- Boxplots
- Correlation Heatmap
- Pairplot
- Model Comparison Chart
- Confusion Matrix
- ROC Curve
- Feature Importance

---

# 📁 Project Structure

```
Heart-Disease-Prediction-ML/
│
├── Heart_Disease_Prediction.ipynb
├── heart.csv
├── heart_disease_logistic_model.pkl
├── requirements.txt
├── README.md
├── LICENSE
└── images/
```


# 📌 Results

- Successfully cleaned and preprocessed the dataset.
- Performed comprehensive Exploratory Data Analysis.
- Compared five Machine Learning algorithms.
- Selected Logistic Regression as the best-performing model.
- Improved performance using Hyperparameter Tuning.
- Saved the final trained model for future predictions.

---

# 🔮 Future Improvements

- Deploy the model using Streamlit or Flask.
- Integrate real-time patient data.
- Train on larger healthcare datasets.
- Explore advanced boosting algorithms such as XGBoost and LightGBM.
- Deploy the solution on cloud platforms.

---

# 👨‍💻 Author

**Hemanth Reddy**

Data Science Student | Machine Learning Enthusiast

GitHub: https://github.com/hemanthreddy2105

---

## ⭐ If you found this project useful, consider giving it a Star!<img width="469" height="542" alt="image" src="https://github.com/user-attachments/assets/820e27ee-797f-4d69-8ada-64cf6a439d04" />
