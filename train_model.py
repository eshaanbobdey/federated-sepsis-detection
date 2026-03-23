import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, roc_auc_score, roc_curve

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import pickle
import os




data = pd.read_csv("data/Dataset.csv")

print("Dataset loaded")




data = data.drop(columns=["Unnamed: 0", "Patient_ID"])




features = [
"HR","O2Sat","Temp","SBP","MAP","DBP","Resp",
"Age","Gender","ICULOS",
"Creatinine","Glucose","Lactate",
"WBC","Platelets","Hgb","Hct","BUN"
]

X = data[features]
y = data["SepsisLabel"]




X = X.fillna(X.median())




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)




scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("SMOTE applied")




model = XGBClassifier(

    max_depth=8,
    learning_rate=0.1,
    n_estimators=600,
    subsample=0.7,
    colsample_bytree=0.8,

    scale_pos_weight=10,
    eval_metric="logloss"

)

print("\nTraining Final XGBoost Model...")

model.fit(X_train, y_train)




probs = model.predict_proba(X_test)[:,1]

threshold = 0.30   

y_pred = (probs > threshold).astype(int)




recall = recall_score(y_test, y_pred)

auc = roc_auc_score(y_test, probs)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

print("Recall:", recall)

print("AUROC:", auc)




fpr, tpr, _ = roc_curve(y_test, probs)

plt.figure()

plt.plot(fpr, tpr, label="XGBoost")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()




importances = model.feature_importances_

plt.figure()

plt.barh(features, importances)

plt.title("Feature Importance")

plt.xlabel("Importance Score")

plt.show()




os.makedirs("models", exist_ok=True)

model_path = "models/sepsis_model.pkl"

pickle.dump(model, open(model_path, "wb"))

print("\nModel saved successfully!")

print(f"Model saved at: {model_path}")