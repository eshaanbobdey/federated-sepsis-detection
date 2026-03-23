import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, roc_auc_score
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input



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



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



scaler = StandardScaler()

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = {
    0: class_weights[0],
    1: class_weights[1]
}

print("Class Weights:", class_weights_dict)



train_data = pd.DataFrame(X_train)
train_data["SepsisLabel"] = y_train.values

train_data = train_data.sample(frac=1, random_state=42)

split1 = int(0.33 * len(train_data))
split2 = int(0.66 * len(train_data))

clients = [
    train_data[:split1],
    train_data[split1:split2],
    train_data[split2:]
]

print("Data split into 3 clients")



def create_model(input_dim, neurons1, neurons2):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(neurons1, activation='relu'),
        Dense(neurons2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model



param_grid = [
    {"neurons1": 32, "neurons2": 16, "epochs": 8},
    {"neurons1": 64, "neurons2": 32, "epochs": 10},
    {"neurons1": 128, "neurons2": 64, "epochs": 10}
]

best_recall = 0
best_model_config = None
best_probs = None

print("\nStarting Hyperparameter Tuning...")



for params in param_grid:

    print(f"\nTesting config: {params}")

    local_models = []

    for client in clients:

        X_local = client.drop(columns=["SepsisLabel"]).values
        y_local = client["SepsisLabel"].values

        model = create_model(
            X_local.shape[1],
            params["neurons1"],
            params["neurons2"]
        )

        model.fit(
            X_local, y_local,
            epochs=params["epochs"],
            batch_size=32,
            verbose=0,
            class_weight=class_weights_dict
        )

        local_models.append(model)

    
    probs = np.zeros(len(X_test))

    for model in local_models:
        probs += model.predict(X_test).flatten()

    probs = probs / len(local_models)

    
    threshold = 0.4
    preds = (probs > threshold).astype(int)

    recall = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print("Recall:", recall)
    print("AUROC:", auc)

    
    if recall > best_recall:
        best_recall = recall
        best_model_config = params
        best_probs = probs



print("\nBest Configuration:", best_model_config)

final_preds = (best_probs > 0.4).astype(int)

print("\nFinal Classification Report:\n")
print(classification_report(y_test, final_preds))

print("Final Recall:", recall_score(y_test, final_preds))
print("Final AUROC:", roc_auc_score(y_test, best_probs))

print("\nFederated Tuned Model Completed!")