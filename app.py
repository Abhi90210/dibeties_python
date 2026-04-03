import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = ["Pregnancies","Glucose","BloodPressure",
            "SkinThickness","Insulin","BMI",
            "DiabetesPedigreeFunction","Age"]

st.title("Diabetes risk predictor")
st.write("Adjust the sliders and click Predict.")

col1, col2 = st.columns(2)
with col1:
    preg = st.slider("Pregnancies", 0, 17, 3)
    gluc = st.slider("Glucose", 0, 200, 120)
    bp   = st.slider("Blood pressure", 0, 122, 70)
    skin = st.slider("Skin thickness", 0, 99, 20)
with col2:
    ins  = st.slider("Insulin", 0, 846, 80)
    bmi  = st.slider("BMI", 0.0, 67.1, 25.0)
    dpf  = st.slider("Diabetes pedigree", 0.08, 2.42, 0.5)
    age  = st.slider("Age", 21, 81, 25)

if st.button("Predict"):
    data = np.array([[preg,gluc,bp,skin,ins,bmi,dpf,age]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    if pred == 1:
        st.error(f"High risk of diabetes — {prob*100:.1f}% confidence")
    else:
        st.success(f"Low risk of diabetes — {(1-prob)*100:.1f}% confidence")

    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(FEATURES, importances)
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance")
    st.pyplot(fig)