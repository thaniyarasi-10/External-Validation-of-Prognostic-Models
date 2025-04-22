import pandas as pd
from lifelines import CoxPHFitter
import joblib

# Sample training dataset (replace with your actual dataset)
data = pd.DataFrame({
    'age': [60, 65, 70, 75, 80],
    'bp': [120, 130, 125, 140, 135],
    'cholesterol': [220, 240, 210, 250, 230],
    'time': [5, 7, 6, 9, 10],  # Duration (survival time)
    'event': [1, 0, 1, 0, 1]   # Event (1=event occurred, 0=censored)
})

# Train a Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(data, duration_col='time', event_col='event')

# Save the trained model to a file
joblib.dump(cph, 'cox_model.pkl')
print("Model has been saved as cox_model.pkl")
