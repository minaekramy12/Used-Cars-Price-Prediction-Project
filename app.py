import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import joblib

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    return joblib.load('models/xgb_pipeline.joblib')

try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception:
    model_loaded = False
    st.warning("⚠️ `xgb_pipeline.joblib` model file not found.")

st.title("🚗 Used Car Price Predictor")
st.markdown("Enter vehicle specifications to estimate the market price using the trained XGBoost model.")
st.divider()

BRANDS = [
    'Ford', 'Hyundai', 'Lexus', 'Infiniti', 'Audi', 'Acura', 'Bmw', 'Tesla',
    'Land', 'Aston', 'Toyota', 'Lincoln', 'Jaguar', 'Mercedes-Benz', 'Dodge',
    'Nissan', 'Genesis', 'Chevrolet', 'Kia', 'Jeep', 'Bentley', 'Honda',
    'Lucid', 'Mini', 'Porsche', 'Hummer', 'Chrysler', 'Volvo', 'Cadillac',
    'Lamborghini', 'Maserati', 'Volkswagen', 'Subaru', 'Rivian', 'Gmc', 'Ram',
    'Alfa', 'Ferrari', 'Scion', 'Mitsubishi', 'Mazda', 'Saturn', 'Bugatti',
    'Polestar', 'Rolls-Royce', 'Mclaren', 'Buick', 'Lotus', 'Pontiac', 'Fiat',
    'Karma', 'Saab', 'Mercury', 'Plymouth', 'Smart', 'Maybach', 'Suzuki'
]

FUEL_TYPES = [
    'E85 Flex Fuel', 'Gasoline', 'Hybrid', 'Unknown', 'Diesel',
    'Plug-In Hybrid', 'not supported'
]

TRANSMISSIONS = ['Automatic', 'Manual', 'CVT', 'Dual-Clutch']
ACCIDENT_OPTIONS = ['None reported', 'At least 1 accident or damage reported']

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Categorical Features")
    
    brand = st.selectbox("Brand", options=BRANDS)
    fuel_type = st.selectbox("Fuel Type", options=FUEL_TYPES)
    transmission = st.selectbox("Transmission", options=TRANSMISSIONS)
    accident = st.selectbox("Accident History", options=ACCIDENT_OPTIONS)

with col2:
    st.subheader("🔢 Numerical Features")
    
    current_year = datetime.now().year
    model_year = st.number_input(
        "Model Year",
        min_value=1990,
        max_value=current_year,
        value=2020,
        step=1
    )
    
    age = current_year - model_year

    milage = st.number_input(
        "Mileage (miles)", 
        min_value=0, 
        value=45000, 
        step=1000
    )
    hp = st.number_input(
        "Horsepower (HP)", 
        min_value=50, 
        max_value=1500, 
        value=180
    )
    engine_size = st.number_input(
        "Engine Size (L)", 
        min_value=0.0, 
        max_value=10.0, 
        value=2.0, 
        step=0.1,
        format="%.1f"
    )
    n_cylinders = st.number_input(
        "Number of Cylinders", 
        min_value=1, 
        max_value=16, 
        value=4, 
        step=1
    )

st.divider()

if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
    
    input_data = pd.DataFrame([{
        'milage': milage,
        'HP': hp,
        'engine_size': engine_size,
        'n_cylinders': n_cylinders,
        'age': age,
        'brand': brand,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'accident': accident
    }])
    
        
    if model_loaded:
        try:
            prediction = pipeline.predict(input_data)[0]
            prediction = np.expm1(prediction)
            st.success("Price calculated successfully!")
            st.metric(label="Predicted Price", value=f"${prediction:,.2f}")
            
        except Exception as err:
            st.error(f"Prediction error: {err}")
    else:
        st.info("Place `xgb_pipeline.joblib` in the directory to run predictions.")