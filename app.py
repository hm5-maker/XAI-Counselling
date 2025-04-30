import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('ensemble_model.pkl')

# Load the label encoders
label_encoders = joblib.load('label_encoders.pkl')

# Set up the title and subheader
st.title("JoSAA Admission Predictor")
st.subheader("Predict your chances of admission based on your inputs")

# Create input fields for each feature
institute = st.selectbox("Select Institute", label_encoders['institute'].classes_)
academic_program_name = st.selectbox("Select Academic Program", label_encoders['academic_program_name'].classes_)
quota = st.selectbox("Select Quota", label_encoders['quota'].classes_)
seat_type = st.selectbox("Select Seat Type", label_encoders['seat_type'].classes_)
gender = st.selectbox("Select Gender", label_encoders['gender'].classes_)

# Add input for opening_rank
opening_rank = st.number_input("Enter Opening Rank", min_value=0, step=1)

# Prepare the input data for prediction
input_data = {
    'institute': label_encoders['institute'].transform([institute])[0],
    'academic_program_name': label_encoders['academic_program_name'].transform([academic_program_name])[0],
    'quota': label_encoders['quota'].transform([quota])[0],
    'seat_type': label_encoders['seat_type'].transform([seat_type])[0],
    'gender': label_encoders['gender'].transform([gender])[0],
    'opening_rank': opening_rank  # Include opening_rank in the input data
}

input_df = pd.DataFrame([input_data])

# Make the prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)

# Display the prediction result
if prediction == 1:  # Assuming 1 indicates admission is likely
    st.success("Congratulations! You are likely to be admitted.")
    st.write(f"Probability of admission: {prediction_proba[0][1]:.2f}")
else:
    st.error("Unfortunately, you are not likely to be admitted.")
    st.write(f"Probability of admission: {prediction_proba[0][0]:.2f}")