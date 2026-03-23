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

client1 = train_data[:split1]
client2 = train_data[split1:split2]
client3 = train_data[split2:]

clients = [client1, client2, client3]

print("Data split into 3 clients (hospitals)")


#FFN
def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model



models = []

print("\nTraining local models...")

for i, client in enumerate(clients):

    X_local = client.drop(columns=["SepsisLabel"]).values
    y_local = client["SepsisLabel"].values

    model = create_model(X_local.shape[1])

    print(f"Training Client {i+1}...")

    model.fit(
        X_local, y_local,
        epochs=8,                
        batch_size=32,
        verbose=0,
        class_weight=class_weights_dict   
    )

    models.append(model)

print("Local training complete")


#FedAVG
print("\nAggregating predictions (Federated)...")

final_probs = np.zeros(len(X_test))

for model in models:
    final_probs += model.predict(X_test).flatten()

final_probs = final_probs / len(models)



threshold = 0.4   
y_pred = (final_probs > threshold).astype(int)



print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, final_probs)

print("Recall:", recall)
print("AUROC:", auc)

print("\nFederated Neural Network Model Completed!")

import pickle

pickle.dump(models, open("models/federated_nn.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Model saved successfully!")