import streamlit as st
import pandas as pd
import xgboost as xgb

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

@st.cache_resource
def load_model():
    m = xgb.XGBRegressor()
    m.load_model("xgb_model.json")
    return m

model = load_model()

default_cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
cols = model.get_booster().feature_names or default_cols

st.title("🏠 House Price Predictor")
st.caption("XGBoost model trained on the Boston Housing dataset. "
           "Academic demo: predictions reflect 1970s data, not current prices.")

left, right = st.columns(2)

with left:
    crim = st.slider("CRIM - Crime rate per capita", 0.0, 90.0, 0.25, 0.01)
    zn = st.slider("ZN - Large residential lots (%)", 0.0, 100.0, 0.0, 0.5)
    indus = st.slider("INDUS - Non-retail business acres (%)", 0.0, 28.0, 9.7, 0.1)
    chas = st.selectbox("CHAS - Borders Charles River (1 = yes)", [0, 1])
    nox = st.slider("NOX - Nitric oxide concentration", 0.38, 0.88, 0.54, 0.01)
    rm = st.slider("RM - Average rooms per dwelling", 3.5, 9.0, 6.2, 0.1)
    age = st.slider("AGE - Homes built before 1940 (%)", 0.0, 100.0, 77.5, 0.5)

with right:
    dis = st.slider("DIS - Distance to employment centres", 1.0, 12.5, 3.2, 0.1)
    rad = st.slider("RAD - Highway accessibility index", 1, 24, 5, 1)
    tax = st.slider("TAX - Property tax rate", 180, 720, 330, 1)
    ptratio = st.slider("PTRATIO - Pupil-teacher ratio", 12.0, 22.5, 19.0, 0.1)
    b = st.slider("B - Demographic index", 0.0, 400.0, 391.4, 0.5)
    lstat = st.slider("LSTAT - Lower-status population (%)", 1.0, 38.0, 11.4, 0.1)

values = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]
row = pd.DataFrame([values], columns=cols).astype(float)

price = float(model.predict(row)[0]) * 1000
st.metric("Estimated median home value", f"${price:,.0f}")
