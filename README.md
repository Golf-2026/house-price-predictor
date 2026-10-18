# House Price Predictor

A machine learning web app that predicts the median home value of a house from 13 neighbourhood features, built with **XGBoost** and **Streamlit**.

** Live demo:** https://house-price-predictor-abhijeet.streamlit.app

## About the project
- **Goal:** predict median house prices from neighbourhood characteristics such as crime rate, number of rooms, tax rate and distance to employment centres.
- **Dataset:** Boston Housing dataset (1970s data, so predictions reflect that era's prices, not today's).
- **Model:** XGBoost Regressor, tuned with hyperparameter search.
- **Result:** R² of **0.9028** on the held-out test set.
- **Front end:** Streamlit app with sliders for each input feature, deployed on Streamlit Community Cloud.

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit front end |
| `xgb_model.json` | Trained XGBoost model |
| `requirements.txt` | Python dependencies |

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack
Python, pandas, scikit-learn, XGBoost, Streamlit

## Note
This is an academic demo. The dataset is from the 1970s, so the predicted prices are not suitable for real-world valuation.# house-price-predictor
House price prediction using XGBoost and Streamlit
