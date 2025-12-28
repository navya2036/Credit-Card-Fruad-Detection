# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions with an interactive Streamlit web interface.

## Project Structure

- `fraud_detection.ipynb` - Main notebook for data exploration and model training
- `app.py` - Streamlit web application for fraud detection
- `requirements.txt` - Python dependencies
- `data/` - Place your dataset here
- `models/` - Saved trained models

## Setup Instructions

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download a credit card fraud dataset (e.g., from Kaggle)

3. Place the dataset in the `data/` folder

4. Open and run `fraud_detection.ipynb` to train the model

5. Launch the Streamlit app:

```bash
streamlit run app.py
```

## Streamlit Web App Features

The interactive web application includes:

- **Single Prediction Mode**: Analyze individual transactions with detailed risk assessment
- **Batch Prediction Mode**: Upload CSV files to analyze multiple transactions at once
- **Model Information**: View model details, feature importance, and performance metrics
- **Visual Analytics**: Interactive charts, gauges, and fraud probability distributions
- **Export Results**: Download flagged transactions for further review

## Dataset Recommendations

- [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Features should include transaction amount, time, and anonymized features
