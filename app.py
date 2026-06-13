# Library heart desease prediction
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from PIL import Image

# Load train model
model = pickle.load(open('model/best_model.pkl', 'rb'))

# Configurate Streamlit app
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Fuction to predict heart disease
#def predict_heart_disease():
st.title("Heart Disease Prediction")
st.write("""
This application predicts the likelihood of heart disease based on various health parameters. Please fill in the details below to get your prediction.
Data obtained from the UCI Machine Learning Repository: [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
""")
st.image("heart-disease.jpg", caption="Heart Disease Prediction", use_container_width=True)

# Input fields for user data
st.sidebar.header("Input Parameters")
st.sidebar.markdown("""Please enter the following health parameters to predict the likelihood of heart disease:
- Age: Age of the patient (years)
- Sex: Gender of the patient (1 = male, 0 = female)
""")

cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
if cp == 0:
    st.sidebar.write("Typical Angina")
elif cp == 1:
    st.sidebar.write("Atypical Angina")
elif cp == 2:
    st.sidebar.write("Non-Anginal Pain")
elif cp == 3:
    st.sidebar.write("Asymptomatic")

thalach = st.sidebar.slider("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)

slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])
if slope == 0:
    st.sidebar.write("Upsloping")
elif slope == 1:
    st.sidebar.write("Flat")
elif slope == 2:
    st.sidebar.write("Downsloping")

oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest (oldpeak)", 0.0, 6.0, 1.0)

exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0, 1])
if exang == 0:
    st.sidebar.write("No")
elif exang == 1:
    st.sidebar.write("Yes")

ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3])
if ca == 0:
    st.sidebar.write("0")
elif ca == 1:
    st.sidebar.write("1")
elif ca == 2:
    st.sidebar.write("2")
elif ca == 3:
    st.sidebar.write("3")

thal = st.sidebar.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
if thal == 0:
    st.sidebar.write("Normal")
elif thal == 1:
    st.sidebar.write("Fixed Defect")
elif thal == 2:
    st.sidebar.write("Reversible Defect")
elif thal == 3:
    st.sidebar.write("Unknown")

sex = st.sidebar.selectbox("Sex", [0, 1])
if sex == 0:
    st.sidebar.write("Female")
elif sex == 1:
    st.sidebar.write("Male")

age = st.sidebar.slider("Age", 20, 100, 50)

# Create a DataFrame for the input data
data = {
    'cp': cp,
    'thalach': thalach,
    'slope': slope,
    'oldpeak': oldpeak,
    'exang': exang,
    'ca': ca,
    'thal': thal,
    'sex': sex,
    'age': age
}
input_data = pd.DataFrame([data])
st.write(input_data)

# Predict heart disease
if st.sidebar.button("Predict"):
    with st.spinner("Predicting..."):
        time.sleep(2)  # Simulate a delay for prediction
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.snow()
            st.error("The model predicts that you have heart disease. Please consult a healthcare professional for further evaluation.")
        else:
            st.balloons()
            st.success("The model predicts that you do not have heart disease. However, please maintain a healthy lifestyle and consult a healthcare professional for regular check-ups.")
