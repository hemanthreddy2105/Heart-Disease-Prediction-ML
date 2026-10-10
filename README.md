# ❤️ Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning project that predicts whether a patient has heart disease from 13 clinical measurements. Five classifiers are compared, a Random Forest is tuned with GridSearchCV, and the final model is saved for reuse.

> ⚠️ **Educational project – not a medical device.** The model is trained on 297 patients from one clinic and must not be used for real diagnosis.

---

## 📊 Results

Five models were trained on an 80/20 stratified split (237 train / 60 test patients) with scaled features where needed:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.917** | 1.000 | 0.821 | 0.902 | 0.953 |
| KNN | 0.900 | 1.000 | 0.786 | 0.880 | 0.959 |
| SVM | 0.900 | 1.000 | 0.786 | 0.880 | 0.940 |
| Random Forest | 0.883 | 0.957 | 0.786 | 0.863 | 0.940 |
| Decision Tree | 0.783 | 0.800 | 0.714 | 0.755 | 0.779 |

A Random Forest was then tuned with `GridSearchCV` (5-fold; best: `n_estimators=300`, `max_depth=5`, `min_samples_leaf=2`, `min_samples_split=2`) → **cross-validated accuracy 0.823, test accuracy 0.85**. Its most important features are `cp`, `thalach`, `ca`, `oldpeak` and `thal`.

**Caveats**

- The test set has only 60 patients (one patient ≈ 1.7 percentage points), so small differences between models are within noise.
- On this split the tuned Random Forest did not beat untuned Logistic Regression; tuning Logistic Regression is a natural next step.
- Recall tops out around 82%, so some patients with heart disease are still missed.

---

## 📂 Dataset

[Cleveland Heart Disease data](https://archive.ics.uci.edu/dataset/45/heart+disease) (UCI Machine Learning Repository): **297 patients, 13 features, no missing values, no duplicates**, 54% healthy / 46% disease.

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG result (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia test result |
| `condition` | **Target** – heart disease present (1) or not (0) |

---

## 🔍 Key findings

- Disease is far more common with **chest-pain type 3**, **exercise-induced angina**, **more blocked vessels (`ca`)** and abnormal **thalassemia** results.
- Patients with disease reach a **lower maximum heart rate** and show higher **ST depression (`oldpeak`)**.
- The tuned Random Forest relies most on `cp`, `thalach`, `ca`, `oldpeak` and `thal`.

---

## 🧪 Method

```
Load & inspect → EDA → stratified 80/20 split → StandardScaler (fit on train only)
      → train 5 classifiers → compare metrics → GridSearchCV on Random Forest
      → feature importance → save model
```

---

## 🚀 Getting started

```bash
git clone https://github.com/hemanthreddy2105/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Re-run the analysis**

```bash
jupyter notebook notebooks/Heart_disease_predicting_analysis.ipynb
```

**Predict for a patient with the saved model**

```bash
python predict.py                                   # built-in example patient
python predict.py --age 40 --sex 0 --cp 1 --thalach 190 --oldpeak 0 --ca 0 --thal 0 --exang 0
```

```python
import joblib, pandas as pd
model = joblib.load("models/heart_disease_model.pkl")
patient = pd.DataFrame([{"age": 62, "sex": 0, "cp": 3, "trestbps": 160, "chol": 164, "fbs": 0,
                         "restecg": 2, "thalach": 145, "exang": 0, "oldpeak": 6.2,
                         "slope": 2, "ca": 3, "thal": 2}])
print(model.predict_proba(patient)[0, 1])           # probability of heart disease
```

> The saved model was created with scikit-learn 1.9; with a very different version, re-run the notebook to regenerate it.

---

## 📁 Project structure

```
Heart-Disease-Prediction-ML/
├── notebooks/
│   └── Heart_disease_predicting_analysis.ipynb   # full analysis with outputs
├── data/
│   └── heart_cleveland_upload.csv                # dataset
├── models/
│   └── heart_disease_model.pkl                   # saved tuned Random Forest
├── predict.py                                    # command-line prediction
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🔮 Future improvements

- Tune Logistic Regression too and pick the final model by cross-validation rather than a single split.
- Use repeated or nested cross-validation for a tighter estimate on this small dataset.
- Tune the decision threshold for higher recall.
- Try gradient boosting (XGBoost / LightGBM) and wrap `predict.py` in a Streamlit app.

---

## 👨‍💻 Author

**Hemanth Reddy** – Data Science student  
GitHub: [@hemanthreddy2105](https://github.com/hemanthreddy2105)

Released under the [MIT License](LICENSE).
