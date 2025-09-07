#Imports
import pandas as pd
import streamlit as st
import joblib
import os

# Load pipeline
model_path = "churn_pipeline_2.pkl"
if os.path.exists(model_path):
    pipeline = joblib.load(model_path)
    st.success("MODEL LOADED SUCCESSFULLY")
else:
    st.warning(f"Model file {model_path} does not exist")

# App Heading
st.title("Customer Churn Prediction")
st.write("Fill your details")
# Form Inputs
with st.form("Churn_Form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("SeniorCitizen", ["Yes", "No"])
    Partner = st.selectbox("Partner", ["Yes","No"])
    Dependents = st.selectbox("Dependents", ["Yes","No"])
    tenure = st.number_input("Tenure (In months)",min_value=0)
    PhoneService = st.selectbox("PhoneService", ["Yes","No"])
    MultipleLines = st.selectbox("MultipleLines", ["Yes","No","No phone service"])
    InternetService = st.selectbox("InternetService", ["DSL","Fiber optic","No"])
    OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes","No","No internet service"])
    OnlineBackup = st.selectbox("OnlineBackup", ["Yes","No","No internet service"])
    DeviceProtection = st.selectbox("DeviceProtection", ["Yes","No","No internet service"])
    TechSupport = st.selectbox("TechSupport", ["Yes","No","No internet service"])
    StreamingTV = st.selectbox("StreamingTV", ["Yes","No","No internet service"])
    StreamingMovies = st.selectbox("StreamingMovies", ["Yes","No","No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes","No"])
    PaymentMethod = st.selectbox("PaymentMethod", ["Bank transfer (automatic)","Credit card (automatic)","Electronic check","Mailed check"])
    MonthlyCharges = st.number_input("MonthlyCharges",min_value=0.0)
    TotalCharges = st.number_input("TotalCharges",min_value=0.0)

    Submitted = st.form_submit_button("Store Now")

# Create DataFrame
if Submitted:
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
        }])
    
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]
    st.success(f"Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Churn Probability : {probability*100:.2f}%")












