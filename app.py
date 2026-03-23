import streamlit as st
from predict import predict_sepsis

st.title("🧠 Sepsis Detection Chatbot")

st.write("Enter patient details:")

# Input fields
HR = st.number_input("Heart Rate")
O2Sat = st.number_input("Oxygen Saturation")
Temp = st.number_input("Temperature")
SBP = st.number_input("Systolic BP")
MAP = st.number_input("MAP")
DBP = st.number_input("Diastolic BP")
Resp = st.number_input("Respiration Rate")
Age = st.number_input("Age")
Gender = st.number_input("Gender (0/1)")
ICULOS = st.number_input("ICU Length of Stay")
Creatinine = st.number_input("Creatinine")
Glucose = st.number_input("Glucose")
Lactate = st.number_input("Lactate")
WBC = st.number_input("WBC")
Platelets = st.number_input("Platelets")
Hgb = st.number_input("Hemoglobin")
Hct = st.number_input("Hematocrit")
BUN = st.number_input("BUN")

if st.button("Predict"):

    input_data = [
        HR, O2Sat, Temp, SBP, MAP, DBP, Resp,
        Age, Gender, ICULOS,
        Creatinine, Glucose, Lactate,
        WBC, Platelets, Hgb, Hct, BUN
    ]

    result = predict_sepsis(input_data)

    st.success(result)

    
