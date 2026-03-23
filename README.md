This project simulates federated learning for privacy-preserving healthcare prediction using decentralized model training.

# Federated Learning-Based Sepsis Detection System

## Overview
This project implements a sepsis detection system using both a centralized machine learning approach and a federated learning-based neural network. The goal is to predict the likelihood of sepsis using patient vital signs and laboratory data while maintaining data privacy.

The project simulates a distributed healthcare environment where multiple clients (hospitals) train models locally without sharing raw data.

---

## Motivation
Sepsis is a critical medical condition that requires early detection. However, healthcare data is sensitive and cannot be freely shared across institutions.

This project focuses on:
- Privacy-preserving machine learning
- Federated learning in healthcare
- Improving recall for early detection

---

## Models Used

### 1. Baseline Model: XGBoost
- Centralized training approach
- Suitable for tabular data
- Captures non-linear relationships effectively

### 2. Federated Model: Feedforward Neural Network
- Multilayer Perceptron (MLP)
- Trained separately on multiple clients
- Aggregates predictions to simulate federated learning

#### Architecture
- Input layer: number of features
- Hidden layer 1: 32 neurons (ReLU)
- Hidden layer 2: 16 neurons (ReLU)
- Output layer: 1 neuron (Sigmoid)

---

## Federated Learning Simulation

The system simulates federated learning using the following steps:

1. Split dataset into training and testing sets
2. Partition training data into multiple clients
3. Train a local model on each client
4. Aggregate predictions from all models
5. Evaluate final performance

---

## Data Preprocessing

- Removed unnecessary columns such as Patient_ID
- Handled missing values using median imputation
- Applied StandardScaler to normalize features

StandardScaler ensures:
- Mean = 0
- Standard deviation = 1

---

## Handling Class Imbalance

The dataset is highly imbalanced:
- Majority class: Non-sepsis
- Minority class: Sepsis

Class weights are used to penalize errors on the minority class.

Loss function becomes:
Loss = class_weight × error

This improves recall for sepsis detection.

---

## Model Training

Each client trains its own neural network:

- Training is done locally
- No data sharing between clients
- Class weights are applied during training

---

## Federated Aggregation

Predictions from all client models are combined:

- Each model predicts probabilities
- Predictions are averaged
- Final probability is used for classification

This simulates federated averaging.

---

## Threshold Tuning

Different thresholds were tested:

- 0.3 → High recall, low accuracy
- 0.4 → Balanced performance

Final threshold used:
0.4

---

## Evaluation Metrics

- Accuracy
- Recall
- AUROC

Recall is prioritized due to the medical nature of the problem.

---

## Results

The model was evaluated using different threshold values to balance recall and accuracy, which is critical in medical diagnosis.

### Threshold = 0.3 (High Sensitivity)

- Recall (Sepsis): ~0.90  
- Accuracy: ~0.33  
- AUROC: ~0.75  

**Observation:**
- Very high recall ensures most sepsis cases are detected  
- However, this leads to a large number of false positives  
- Not ideal for practical deployment due to low precision  

---

### Threshold = 0.4 (Balanced Performance)

- Recall (Sepsis): ~0.73–0.74  
- Accuracy: ~0.62  
- AUROC: ~0.76  

**Observation:**
- Provides a better balance between recall and accuracy  
- Reduces false positives while still detecting most sepsis cases  
- Selected as the final operating threshold  

---

### Class-wise Performance (Approximate)

| Class | Precision | Recall | F1-Score | Support |
|------|----------|--------|----------|--------|
| 0 (Non-Sepsis) | High (~0.99) | Moderate (~0.62) | Good | Large |
| 1 (Sepsis)     | Low (~0.03)  | High (~0.73)     | Low  | Small |

---

### Key Insights

- The dataset is highly imbalanced, which affects precision for the minority class  
- Class weighting significantly improves recall for sepsis detection  
- Threshold tuning plays a crucial role in balancing sensitivity and specificity  
- AUROC (~0.76) indicates good separability between classes  

---

### Final Model Selection

The model with **threshold = 0.4** was selected because:

- It maintains high recall for sepsis detection  
- It significantly reduces false positives compared to lower thresholds  
- It provides a practical balance for real-world healthcare deployment  

---

## How to Run

### 1. Create and Activate Virtual Environment

#### On macOS/Linux:
```bash
python3 -m venv fl_env
source fl_env/bin/activate
```

#### On Windows:
```bash
python -m venv fl_env
fl_env\Scripts\activate
```

---

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 3. Train the Model
```bash
python federated_model.py
```

---

### 4. Run the Application
```bash
streamlit run app.py
```

Then open your browser and go to:
```bash
http://localhost:8501
```

---

## Key Concepts Demonstrated

- Federated learning simulation
- Neural networks for tabular data
- Handling class imbalance
- Threshold tuning
- Model evaluation

---

## Privacy Considerations

This approach aligns with healthcare data regulations such as:
- HIPAA
- GDPR

Federated learning ensures that raw data is not shared between clients.

---

## Future Improvements

- Implement weight-based aggregation (true FedAvg)
- Add FedProx algorithm
- Deploy real distributed federated system
- Improve precision

---

## Author

Eshaan Tushar Bobdey
