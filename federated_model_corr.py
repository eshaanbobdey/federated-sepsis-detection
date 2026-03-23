import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, roc_auc_score
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input



data = pd.read_csv("data/Dataset.csv")
print("Dataset loaded")


data = data.drop(columns=["Unnamed: 0", "Patient_ID"])



corr_matrix = data.corr()


plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix")
plt.show()



target_corr = corr_matrix["SepsisLabel"].abs().sort_values(ascending=False)

print("\nFeature Correlation with Target:\n")
print(target_corr)


threshold = 0.05   

selected_features = target_corr[target_corr > threshold].index.tolist()


selected_features.remove("SepsisLabel")

print("\nSelected Features:\n", selected_features)



X = data[selected_features]
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



model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)



model.fit(
    X_train, y_train,
    epochs=8,
    batch_size=32,
    verbose=1,
    class_weight=class_weights_dict
)



probs = model.predict(X_test).flatten()

threshold = 0.4
y_pred = (probs > threshold).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))
print("AUROC:", roc_auc_score(y_test, probs))