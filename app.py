# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import shap
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Admission Predictor with XAI", layout="centered")
st.title("🎓 Admission Predictor with SHAP Explanation")

# Load label encoders
label_encoders = joblib.load("models/label_encoders.pkl")
categorical_columns = ['institute', 'academic_program_name', 'quota', 'seat_type', 'gender']

# Collect user input
st.header("Step 1: Enter Candidate Details")
input_data = {}

for col in categorical_columns:
    options = label_encoders[col].classes_
    user_val = st.selectbox(f"{col.replace('_', ' ').title()}", options)
    input_data[col] = label_encoders[col].transform([user_val])[0]

input_data['opening_rank'] = st.number_input("Opening Rank", min_value=1, max_value=200000, value=10000)

# Model selection
st.header("Step 2: Select Model")
model_name = st.selectbox("Choose Model", [
    "logistic_regression", "random_forest", "svc", "voting_classifier"
])
model = joblib.load(f"models/{model_name}.pkl")

# Predict and explain
st.header("Step 3: Prediction and Explanation")
if st.button("Predict Admission"):
    user_df = pd.DataFrame([input_data])
    background = pd.read_excel("cleaned_josaa_data (1).xlsx")

    # Encode background
    for col in categorical_columns:
        le = LabelEncoder()
        background[col] = le.fit_transform(background[col])

    background = background.drop(columns=['year', 'round', 'closing_rank'])
    background_sample = background.sample(100, random_state=42)

    # Prediction
    prediction = model.predict(user_df)[0]
    proba = model.predict_proba(user_df)[0][1]
    result = "✅ Admitted" if prediction == 1 else "❌ Not Admitted"
    st.success(f"Prediction: {result} | Probability: {proba:.2f}")

    # Choose SHAP explainer
    if model_name == "random_forest":
        explainer = shap.TreeExplainer(model)
    elif model_name == "logistic_regression":
        explainer = shap.Explainer(model, background_sample)
    else:
        explainer = shap.Explainer(model.predict_proba, background_sample)

    shap_values = explainer(user_df)

    # Explanation object (fix for multi-class)
    explanation = shap.Explanation(
        values=shap_values.values[0, 1] if shap_values.values.ndim == 3 else shap_values.values[0],
        base_values=shap_values.base_values[0, 1] if hasattr(shap_values.base_values[0], '__len__') else shap_values.base_values[0],
        data=user_df.values[0],
        feature_names=user_df.columns
    )

    # Plot SHAP Waterfall using matplotlib-compatible method
    st.subheader("🔍 SHAP Explanation (Waterfall Plot)")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    st.pyplot(fig)

    st.caption("Feature impacts are shown in red (negative) and blue (positive). Higher blue = more likely to be admitted.")
