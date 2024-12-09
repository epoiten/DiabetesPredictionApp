import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your trained model and label encoder
with open('simple_DP_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('simple_DP_label_encoder.pkl', 'rb') as f:
    le_class = pickle.load(f)

# Title of the web app
st.title("🩺Diabetes Prediction Web App💉")
st.subheader("ReDI School - Hamburg : Machine Learning Class Project")
st.markdown("**Model:** K-Nearest Neighbor(KNN) Algorithm with F1 Score of 0.87")
st.markdown(
    """
    **⚠️ Disclaimer:**  
    This app is for informational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. Predictions may not always be accurate. If you have health concerns, please consult a healthcare professional.
    """,
    unsafe_allow_html=True
)
st.markdown("Provide your health metrics to predict if you are at risk of diabetes. Please ensure values are accurate for the best results.")
# Input fields for Age, Sex, Weight, and Height
st.subheader("Personal and Medical Information")
gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
age = st.slider("Age (years)", min_value=20, max_value=79, value=30, help="Enter your age.")
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1, help="Enter your weight in kilograms.")
height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.1, help="Enter your height in centimeters.")

# Calculate BMI from Weight and Height
bmi = weight / ((height / 100) ** 2)

# Display calculated BMI
st.write(f"Calculated BMI: {bmi:.2f}")

# Encoding and prediction
gender_encoded = 1 if gender == "Male" else 0
custom_input = [gender_encoded, age, bmi]

if st.button("Predict"):
    custom_input_array = np.array(custom_input).reshape(1, -1)
    prediction = model.predict(custom_input_array)
    predicted_class = le_class.inverse_transform(prediction)[0]

    # Display results
    if predicted_class == 'Y':
        st.warning("⚠️ You have a high chance of diabetes. Please consult a doctor.")
    elif predicted_class == 'N':
        st.success("✅ Good news! You have a low chance of diabetes. Maintain a healthy lifestyle.")
    elif predicted_class == 'P':
        st.info("⚠️ You may be prediabetic. Monitor your sugar intake and consult a doctor.")
    else:
        st.error("Unexpected result. Please re-check your inputs.")
