This project simulates federated learning for privacy-preserving healthcare prediction using decentralized model training.

# Federated Learning-Based Sepsis Detection System

## Overview
This project implements a sepsis detection system using both a centralized machine learning approach (XGBoost) and a federated-learning-*style* neural network. The goal is to predict the likelihood of sepsis using patient vital signs and laboratory data while exploring privacy-preserving training patterns.

The project simulates a distributed healthcare environment: multiple "client" models are trained on partitions of a single dataset within this codebase, and their predictions are combined. It is a single-machine simulation, not a live multi-institution deployment — see [Relationship to `federated_sepsis_website`](#relationship-to-federated_sepsis_website) below for the sibling project that implements the networked side of this.

> **Dataset:** [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) — early sepsis prediction from ICU vitals and labs.

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

### 2. Federated-Style Model: Feedforward Neural Network
- Multilayer Perceptron (MLP)
- Trained separately on multiple simulated clients
- **Aggregation method: prediction averaging**, not weight averaging — each client model outputs its own probability, and those probabilities are averaged to produce the final prediction. This is a simpler approximation of federated learning; true FedAvg averages model *weights*, not outputs (see Future Improvements).

#### Architecture
- Input layer: number of features (see note above on specifying the dataset/feature count)
- Hidden layer 1: 32 neurons (ReLU)
- Hidden layer 2: 16 neurons (ReLU)
- Output layer: 1 neuron (Sigmoid)

---

## Federated Learning Simulation

The system simulates federated learning using the following steps, all executed within a single script/process:

1. Split dataset into training and testing sets
2. Partition training data into multiple clients
3. Train a local model on each client
4. Aggregate predictions from all models
5. Evaluate final performance

No network communication, per-client authentication, or independent client execution is involved — that infrastructure lives in the separate `federated_sepsis_website` repo.

---

## Data Preprocessing

- Removed unnecessary columns such as `Patient_ID`
- Handled missing values using median imputation
- Applied `StandardScaler` to normalize features

`StandardScaler` ensures:
- Mean = 0
- Standard deviation = 1

---

## Handling Class Imbalance

The dataset is highly imbalanced:
- Majority class: Non-sepsis
- Minority class: Sepsis

Class weights are used to penalize errors on the minority class:

```
Loss = class_weight × error
```

This improves recall for sepsis detection, at a precision cost — see Results below.

---

## Model Training

Each simulated client trains its own neural network:
- Training is done on that client's data partition
- No data sharing between clients within the simulation
- Class weights are applied during training

---

## Federated Aggregation

Predictions from all client models are combined:
- Each model predicts probabilities
- Predictions are averaged
- Final probability is used for classification

This approximates the *effect* of federated averaging without implementing FedAvg's weight-averaging mechanism.

---

## Threshold Tuning

Different thresholds were tested:
- 0.3 → high recall, low accuracy
- 0.4 → balanced performance (final choice)

---

## Evaluation Metrics

- Accuracy
- Recall
- AUROC

Recall is prioritized due to the medical nature of the problem: a missed sepsis case is generally costlier than a false alarm.

---

## Results

Evaluated on the [PhysioNet 2019 Challenge](https://physionet.org/content/challenge-2019/1.0.0/) dataset. *(Train/test split size and any train/test split random seed are still worth adding here for full reproducibility.)*

### Threshold = 0.3 (High Sensitivity)
- Recall (Sepsis): ~0.90
- Accuracy: ~0.33
- AUROC: ~0.75

Very high recall catches most sepsis cases, but at the cost of a large number of false positives — accuracy this low means the model is flagging non-sepsis patients constantly. Not viable for practical deployment as-is.

### Threshold = 0.4 (Balanced Performance — final choice)
- Recall (Sepsis): ~0.73–0.74
- Accuracy: ~0.62
- AUROC: ~0.76

Better balance, though accuracy (~0.62) and sepsis-class precision (below) are still far from deployment-ready.

### Class-wise Performance (Approximate, threshold = 0.4)

| Class          | Precision    | Recall           | F1-Score | Support |
| -------------- | ------------ | ---------------- | -------- | ------- |
| 0 (Non-Sepsis) | ~0.99        | ~0.62            | Good     | Large   |
| 1 (Sepsis)     | ~0.03        | ~0.73            | Low      | Small   |

**Worth calling out directly:** ~0.03 precision on the sepsis class means the large majority of "sepsis" alerts at this threshold are false alarms — for every real sepsis case correctly flagged, roughly 30+ non-sepsis patients are also flagged. That's a known, expected consequence of prioritizing recall on a severely imbalanced dataset with a low decision threshold, not a bug — but it also means this model is not close to clinically usable in its current form. Improving sepsis-class precision (via better features, resampling, or a cost-sensitive threshold search) is a bigger open problem than the "Final Model Selection" framing below might suggest on its own.

### Key Insights
- The dataset is highly imbalanced, which severely limits precision for the minority (sepsis) class
- Class weighting improves recall for sepsis detection, at a large precision cost
- Threshold tuning trades recall against both accuracy and precision
- AUROC (~0.76) indicates fair, not strong, separability between classes

### Final Model Selection
Threshold = 0.4 was selected as a working default because it improves accuracy and reduces false-positive volume relative to threshold 0.3, while retaining most of the recall. It is a reasonable starting point for further tuning, not a deployment-ready operating point — see class-wise precision above.

---

## Relationship to `federated_sepsis_website`

This repo and **[federated_sepsis_website](https://github.com/eshaanbobdey/federated_sepsis_website)** are companion projects that aren't yet integrated:

- **This repo** proves the ML approach on real data — training, class-imbalance handling, threshold tuning, and evaluation — but its "federated" simulation runs on one machine and aggregates client *predictions*, not model *weights*.
- **`federated_sepsis_website`** is the reverse: real infrastructure (hospital auth, upload API, versioned aggregation) implementing true FedAvg weight-averaging across a network — but it has no training pipeline, so the weights it aggregates are currently random placeholders, not weights learned here.

Connecting them — exporting a model trained here into the `.pkl` weight format `federated_sepsis_website`'s upload endpoint expects — is listed as a future improvement in both READMEs.

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

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python federated_model.py
```

### 4. Run the Application
```bash
streamlit run app.py
```

Then open your browser and go to:
```
http://localhost:8501
```

---

## Key Concepts Demonstrated
- Federated-learning-style simulation (prediction averaging)
- Neural networks for tabular data
- Handling class imbalance
- Threshold tuning
- Model evaluation

---

## Privacy Considerations

The simulated approach avoids centralizing raw data across clients within this codebase, which is conceptually aligned with the goals of regulations like HIPAA and GDPR. Note this repo doesn't itself implement any compliance controls (access logging, encryption, data retention policy, etc.) — treat this as a research prototype demonstrating the ML technique, not a compliant system.

---

## Future Improvements
- Implement weight-based aggregation (true FedAvg), ideally by connecting to `federated_sepsis_website`'s aggregation API
- Add FedProx algorithm
- Deploy a real distributed federated system (multiple independent processes/machines)
- Improve sepsis-class precision at the chosen operating threshold
- Document exact train/test split size and random seed for full reproducibility

---

## Author

Eshaan Tushar Bobdey
2. Partition training data into multiple clients
3. Train a local model on each client
4. Aggregate predictions from all models
5. Evaluate final performance

No network communication, per-client authentication, or independent client execution is involved — that infrastructure lives in the separate `federated_sepsis_website` repo.

---

## Data Preprocessing

- Removed unnecessary columns such as `Patient_ID`
- Handled missing values using median imputation
- Applied `StandardScaler` to normalize features

`StandardScaler` ensures:
- Mean = 0
- Standard deviation = 1

---

## Handling Class Imbalance

The dataset is highly imbalanced:
- Majority class: Non-sepsis
- Minority class: Sepsis

Class weights are used to penalize errors on the minority class:

```
Loss = class_weight × error
```

This improves recall for sepsis detection, at a precision cost — see Results below.

---

## Model Training

Each simulated client trains its own neural network:
- Training is done on that client's data partition
- No data sharing between clients within the simulation
- Class weights are applied during training

---

## Federated Aggregation

Predictions from all client models are combined:
- Each model predicts probabilities
- Predictions are averaged
- Final probability is used for classification

This approximates the *effect* of federated averaging without implementing FedAvg's weight-averaging mechanism.

---

## Threshold Tuning

Different thresholds were tested:
- 0.3 → high recall, low accuracy
- 0.4 → balanced performance (final choice)

---

## Evaluation Metrics

- Accuracy
- Recall
- AUROC

Recall is prioritized due to the medical nature of the problem: a missed sepsis case is generally costlier than a false alarm.

---

## Results

*(Pending: dataset name/version and train/test split size — add here for reproducibility.)*

### Threshold = 0.3 (High Sensitivity)
- Recall (Sepsis): ~0.90
- Accuracy: ~0.33
- AUROC: ~0.75

Very high recall catches most sepsis cases, but at the cost of a large number of false positives — accuracy this low means the model is flagging non-sepsis patients constantly. Not viable for practical deployment as-is.

### Threshold = 0.4 (Balanced Performance — final choice)
- Recall (Sepsis): ~0.73–0.74
- Accuracy: ~0.62
- AUROC: ~0.76

Better balance, though accuracy (~0.62) and sepsis-class precision (below) are still far from deployment-ready.

### Class-wise Performance (Approximate, threshold = 0.4)

| Class          | Precision    | Recall           | F1-Score | Support |
| -------------- | ------------ | ---------------- | -------- | ------- |
| 0 (Non-Sepsis) | ~0.99        | ~0.62            | Good     | Large   |
| 1 (Sepsis)     | ~0.03        | ~0.73            | Low      | Small   |

**Worth calling out directly:** ~0.03 precision on the sepsis class means the large majority of "sepsis" alerts at this threshold are false alarms — for every real sepsis case correctly flagged, roughly 30+ non-sepsis patients are also flagged. That's a known, expected consequence of prioritizing recall on a severely imbalanced dataset with a low decision threshold, not a bug — but it also means this model is not close to clinically usable in its current form. Improving sepsis-class precision (via better features, resampling, or a cost-sensitive threshold search) is a bigger open problem than the "Final Model Selection" framing below might suggest on its own.

### Key Insights
- The dataset is highly imbalanced, which severely limits precision for the minority (sepsis) class
- Class weighting improves recall for sepsis detection, at a large precision cost
- Threshold tuning trades recall against both accuracy and precision
- AUROC (~0.76) indicates fair, not strong, separability between classes

### Final Model Selection
Threshold = 0.4 was selected as a working default because it improves accuracy and reduces false-positive volume relative to threshold 0.3, while retaining most of the recall. It is a reasonable starting point for further tuning, not a deployment-ready operating point — see class-wise precision above.

---

## Relationship to `federated_sepsis_website`

This repo and **[federated_sepsis_website](https://github.com/eshaanbobdey/federated_sepsis_website)** are companion projects that aren't yet integrated:

- **This repo** proves the ML approach on real data — training, class-imbalance handling, threshold tuning, and evaluation — but its "federated" simulation runs on one machine and aggregates client *predictions*, not model *weights*.
- **`federated_sepsis_website`** is the reverse: real infrastructure (hospital auth, upload API, versioned aggregation) implementing true FedAvg weight-averaging across a network — but it has no training pipeline, so the weights it aggregates are currently random placeholders, not weights learned here.

Connecting them — exporting a model trained here into the `.pkl` weight format `federated_sepsis_website`'s upload endpoint expects — is listed as a future improvement in both READMEs.

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

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python federated_model.py
```

### 4. Run the Application
```bash
streamlit run app.py
```

Then open your browser and go to:
```
http://localhost:8501
```

---

## Key Concepts Demonstrated
- Federated-learning-style simulation (prediction averaging)
- Neural networks for tabular data
- Handling class imbalance
- Threshold tuning
- Model evaluation

---

## Privacy Considerations

The simulated approach avoids centralizing raw data across clients within this codebase, which is conceptually aligned with the goals of regulations like HIPAA and GDPR. Note this repo doesn't itself implement any compliance controls (access logging, encryption, data retention policy, etc.) — treat this as a research prototype demonstrating the ML technique, not a compliant system.

---

## Future Improvements
- Implement weight-based aggregation (true FedAvg), ideally by connecting to `federated_sepsis_website`'s aggregation API
- Add FedProx algorithm
- Deploy a real distributed federated system (multiple independent processes/machines)
- Improve sepsis-class precision at the chosen operating threshold
- Document the dataset source, version, and train/test split sizes used for the Results section above

---

## Author

Eshaan Tushar Bobdey
