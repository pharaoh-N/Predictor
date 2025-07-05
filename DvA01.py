import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(url, header=None, names=columns)
    features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
    X = df[features].values
    y = df["Outcome"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler

model, scaler = load_model()

st.title("Diabetes Prediction App")
st.markdown("Enter your health data below to estimate the probability of diabetes.")

pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
height_m = height_cm / 100
bmi = weight / (height_m ** 2)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, bmi, age]])
    input_scaled = scaler.transform(input_data)

    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")
    st.write(f"**Predicted Probability of Diabetes:** {probability:.2f}")
    st.write(f"**Diagnosis:** {'You likely have diabetes' if prediction == 1 else 'Unlikely to have diabetes'}")

    # Optional plot
    st.subheader("Effect of Glucose on Prediction (Other Inputs Fixed)")
    glucose_range = np.linspace(40, 200, 300)
    example_inputs = np.tile(input_data, (300, 1))
    example_inputs[:, 1] = glucose_range  # Vary glucose
    example_scaled = scaler.transform(example_inputs)
    probs = model.predict_proba(example_scaled)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(glucose_range, probs, label="Probability Curve", color='blue')
    ax.axvline(glucose, color='red', linestyle='--', label="Your Glucose Level")
    ax.set_xlabel("Glucose Level")
    ax.set_ylabel("Probability of Diabetes")
    ax.set_title("Probability vs. Glucose (Other Inputs Fixed)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
