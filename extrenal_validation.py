import pandas as pd
import numpy as np
import joblib
from lifelines.utils import concordance_index
from sklearn.metrics import brier_score_loss

# Load trained Cox model
cox_model = joblib.load("cox_model.pkl")

# Load external dataset
external_data = pd.read_csv("external_dataset.csv")

# Column names
duration_col = 'time'   # Survival duration column
event_col = 'event'     # Event indicator column (1=event occurred, 0=censored)

# Match features used in training
X_test = external_data[cox_model.params_.index]
y_test_duration = external_data[duration_col]
y_test_event = external_data[event_col]

# Predict risk scores
risk_scores = cox_model.predict_partial_hazard(X_test)

# Calculate Concordance Index (C-index)
c_index = concordance_index(y_test_duration, -risk_scores, y_test_event)
print(f"Concordance Index (C-index): {c_index:.4f}")

# Select time point for Brier Score (median survival time)
time_point = np.percentile(y_test_duration, 50)

# Predict survival probabilities at time_point
surv_probs = cox_model.predict_survival_function(X_test, times=[time_point]).T.squeeze()

# Create binary outcomes for Brier Score
event_observed = (y_test_duration <= time_point) & (y_test_event == 1)

# Calculate Brier Score
brier = brier_score_loss(event_observed, 1 - surv_probs)
print(f"Brier Score at time {time_point:.2f}: {brier:.4f}")
