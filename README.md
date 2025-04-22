# External Validation of Prognostic Model

This project provides an implementation to perform **external validation** of a Cox Proportional Hazards model on new datasets. The model was trained and saved using a Cox model, and this project allows you to evaluate its performance on an external dataset.

## Contents

- `external_validation.py`: The main script for performing the external validation.
- `train_and_save_model.py`: A script to train the Cox Proportional Hazards model and save it as `cox_model.pkl`.
- `external_dataset.csv`: An external dataset for testing the trained Cox model. Replace this with your own dataset.
- `requirements.txt`: A list of Python libraries required to run the code.
- `README.md`: This documentation file.

## Prerequisites

Before running this project, ensure you have the following Python libraries installed:

- `pandas`
- `lifelines`
- `scikit-learn`
- `joblib`
- `numpy`

You can install all dependencies by running:

```bash
pip install -r requirements.txt
