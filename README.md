# Diabetes Risk Predictor

A machine learning web app that predicts the risk of diabetes based on health parameters. Built with scikit-learn and deployed using Streamlit.

🔗 **Live Demo**: [https://dibetiespython-82hdgqmchqprfvvad94hbq.streamlit.app/](https://dibetiespython-82hdgqmchqprfvvad94hbq.streamlit.app/)

---

## About the project

This project uses the Pima Indians Diabetes Dataset to train a Random Forest classifier that predicts whether a patient is at risk of diabetes based on 8 health features. The model is served through an interactive Streamlit web app where users can adjust input sliders and get a real-time prediction with a confidence score and feature importance chart.

---

## Tech stack

- **Python** — core language
- **pandas & numpy** — data loading, cleaning, and EDA
- **scikit-learn** — model training (Random Forest), scaling, evaluation
- **matplotlib & seaborn** — data visualization
- **Streamlit** — interactive web app
- **joblib** — model serialization
- **Streamlit Cloud** — free deployment

---

## Model performance

| Metric | Score |
|--------|-------|
| Accuracy | 74% |
| Precision (no diabetes) | 0.78 |
| Recall (no diabetes) | 0.83 |
| Precision (diabetes) | 0.65 |
| Recall (diabetes) | 0.57 |

---

## Key findings from EDA

- Dataset has 768 rows and 8 features
- Class imbalance: 65% no diabetes vs 35% diabetes — handled using stratified split and `class_weight='balanced'`
- Columns like Glucose, Insulin, BMI had impossible zero values — replaced with column medians
- Glucose and BMI are the most important features for prediction

---

## How to run locally

```bash
# 1. Clone the repo
git clone https://github.com/Abhi90210/dibeties_python.git
cd dibeties_python

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python training_model.py

# 4. Run the app
streamlit run app.py
```

---

## Project structure

```
dibeties_python/
├── app.py                  # Streamlit web app
├── training_model.py       # Model training script
├── eda.ipynb               # Exploratory data analysis notebook
├── diabetes.csv            # Dataset
├── model.pkl               # Saved Random Forest model
├── scaler.pkl              # Saved StandardScaler
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## What I learned

- How to handle missing values disguised as zeros in medical datasets
- How to address class imbalance using stratified splits and balanced class weights
- How to evaluate a classifier using precision, recall, and confusion matrix
- How to deploy a machine learning model as a live web app using Streamlit Cloud

---

*Built by Abhi90210 — 2nd year Electronics and Computer Science student, Pune*