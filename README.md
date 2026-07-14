#  Heart Disease Prediction using Machine Learning

An end-to-end machine learning project that predicts the presence of heart disease in a
patient from clinical parameters. Built as part of the **Edunet Foundation / IBM SkillsBuild
AI & ML Internship**.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo%20App-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

##  Project Overview

Cardiovascular disease is one of the leading causes of death worldwide. This project uses
supervised machine learning to predict whether a patient is likely to have heart disease
based on 13 routinely-collected clinical features (age, chest pain type, cholesterol,
resting blood pressure, ECG results, etc.).

Seven classification algorithms are trained and compared, the best model is
hyperparameter-tuned, and the final pipeline is packaged into an interactive **Streamlit web
app** for live predictions.

##  Objectives

- Perform exploratory data analysis (EDA) to understand feature relationships with heart disease
- Preprocess and scale clinical data for machine learning
- Train and compare multiple classification models
- Tune the best-performing model with `GridSearchCV`
- Evaluate using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Deploy the final model as an interactive demo app

## Dataset

**Source:** [UCI Machine Learning Repository — Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- 303 patient records, 13 clinical features, 1 binary target
- No missing values

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) |
| `target` | 1 = heart disease present, 0 = absent |

##  Models Trained

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Support Vector Machine
- Naive Bayes

##  Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV Accuracy |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.820 | 0.762 | 0.970 | **0.853** | **0.912** | 0.835 |
| Support Vector Machine | 0.820 | 0.775 | 0.939 | 0.849 | 0.883 | 0.802 |
| Gradient Boosting | 0.820 | 0.789 | 0.909 | 0.845 | 0.879 | 0.802 |
| K-Nearest Neighbors | 0.820 | 0.789 | 0.909 | 0.845 | 0.884 | 0.810 |
| Naive Bayes | 0.820 | 0.789 | 0.909 | 0.845 | 0.876 | 0.806 |
| Logistic Regression | 0.803 | 0.769 | 0.909 | 0.833 | 0.869 | 0.831 |
| Decision Tree | 0.705 | 0.703 | 0.788 | 0.743 | 0.697 | 0.732 |

**Best model:** Random Forest (tuned via GridSearchCV) — final test ROC-AUC of **0.921**.

> Full numeric results are also saved to `models/model_comparison_results.csv` after running the training script.

### Visualizations

| Target Distribution | Correlation Heatmap |
|---|---|
| ![target](images/target_distribution.png) | ![corr](images/correlation_heatmap.png) |

| Model Comparison | ROC Curves |
|---|---|
| ![comparison](images/model_comparison.png) | ![roc](images/roc_curves.png) |

| Confusion Matrix (Best Model) |
|---|
| ![cm](images/confusion_matrix_best_model.png) |

##  Project Structure

```
heart-disease-prediction/
├── app/
│   └── app.py                     # Streamlit demo app
├── data/
│   └── heart.csv                  # UCI Heart Disease dataset
├── images/                        # EDA & evaluation plots (generated)
├── models/                        # Saved model, scaler & metadata (generated)
├── notebooks/
│   └── Heart_Disease_Prediction.ipynb   # Full analysis notebook, with outputs
├── src/
│   └── train.py                   # End-to-end training pipeline script
├── requirements.txt
├── LICENSE
└── README.md
```



###  Explore the notebook
Open `notebooks/Heart_Disease_Prediction.ipynb` in Jupyter to see the full step-by-step
analysis with embedded outputs and visualizations.

###  Run the demo app
```bash
streamlit run app/app.py
```
This launches an interactive web UI where you can enter patient details and get a live
prediction with probability score.

##  Tech Stack

- **Language:** Python 3.10+
- **Data handling:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Deployment demo:** Streamlit
- **Model persistence:** joblib

##  Future Improvements

- Add SHAP-based model explainability
- Try stacking/ensembling top models
- Deploy the Streamlit app publicly (e.g., Streamlit Community Cloud)
- Expand the dataset with more diverse patient populations



---

*This project was developed as part of an academic internship and is intended for
educational purposes only. It is not a certified medical diagnostic tool.*
