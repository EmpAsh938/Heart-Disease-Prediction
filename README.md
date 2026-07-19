# Heart Disease Prediction

## Objective

Predict whether a patient has heart disease using machine learning algorithms.

## Dataset

(UCI Heart Disease Dataset)

## Status

Project setup completed and the notebook workflow has been implemented as reusable source modules.

## Run the training pipeline

```bash
.venv/bin/python train.py --model random_forest --output-dir models
```

## Run inference

```bash
.venv/bin/python predict.py --model models/random_forest_model.joblib --input data/raw/heart_disease_data.csv --features models/feature_columns.txt
```

## Validate the implementation

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Launch the Streamlit app

```bash
.venv/bin/streamlit run app.py
```

This will start an interactive web interface where you can:

- Make single patient predictions
- Upload CSV files for batch predictions
- View model metrics and performance
- Download prediction results
