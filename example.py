import xgboost
print(xgboost.__version__)

import pickle

pickle.dump(models, open("models/federated_nn.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Model saved successfully!")