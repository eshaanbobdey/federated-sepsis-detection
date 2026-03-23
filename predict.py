import numpy as np
import pickle

# Load models + scaler
models = pickle.load(open("models/federated_nn.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Features order (VERY IMPORTANT)
features = [
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp",
    "Age","Gender","ICULOS",
    "Creatinine","Glucose","Lactate",
    "WBC","Platelets","Hgb","Hct","BUN"
]

def predict_sepsis(input_data):

    input_array = np.array(input_data).reshape(1, -1)
    input_array = scaler.transform(input_array)

    # Federated prediction
    probs = 0
    for model in models:
        probs += model.predict(input_array)[0][0]

    probs = probs / len(models)

    # Threshold
    if probs > 0.4:
        return f"⚠️ High Risk of Sepsis (Probability: {probs:.2f})"
    else:
        return f"✅ Low Risk (Probability: {probs:.2f})"