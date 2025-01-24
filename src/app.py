import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

# Título de la aplicación
st.title("Predicción de Retrasos en Vuelos")

# Cargar el modelo preentrenado
@st.cache_resource
def load_model():
    with open("/workspace/final_project/models/best_model_xgb_subsample_1.0_n_estimators_200_max_depth_10_learning_rate_0.2_gamma_0.1_colsample_bytree_0.8.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

with open('data/interim/enc_Airline.json', 'r') as f:
    airline_classes = json.load(f)

with open('data/interim/enc_Dest.json', 'r') as f:
    dest_classes = json.load(f)

with open('data/interim/enc_DestCityName.json', 'r') as f:
    dest_city_classes = json.load(f)

with open('data/interim/enc_DestStateName.json', 'r') as f:
    dest_state_classes = json.load(f)

with open('data/interim/enc_Origin.json', 'r') as f:
    origin_classes = json.load(f)
    
with open('data/interim/enc_OriginCityName.json', 'r') as f:
    origin_city_classes = json.load(f)

with open('data/interim/enc_OriginStateName.json', 'r') as f:
    origin_state_classes = json.load(f)

# Columnas utilizadas en el modelo
features = [
    "Airline", "Origin", "Dest", "OriginCityName", "DestCityName", "OriginStateName", "DestStateName", "CRSDepTime", "CRSArrTime",
    "Distance", "Quarter", "Month", "DayofMonth", "DayOfWeek", "WeekType", "dia_festivo"
]

# Función para preprocesar los datos
@st.cache_data
def preprocess_data(input_data, _encoders, _scaler):  # Renombrar parámetros para evitar problemas de hashing
    # Codificar características categóricas
    for col in _encoders:
        if col in input_data:
            input_data[col] = _encoders[col].transform(input_data[col])
    
    # Escalar características numéricas
    numeric_cols = ["CRSDepTime", "CRSArrTime", "Distance", "Quarter", "Month", "DayofMonth", "DayOfWeek"]
    input_data[numeric_cols] = _scaler.transform(input_data[numeric_cols])
    
    return input_data

# Entradas del usuario
st.sidebar.header("Introducir características del vuelo")

airline = st.sidebar.selectbox("Aerolínea", airline_classes)
origin = st.sidebar.text_input("Código de aeropuerto de origen", origin_classes)
dest = st.sidebar.text_input("Código de aeropuerto de destino", dest_classes)
origin_city = st.sidebar.selectbox("Ciudad de origen", origin_city_classes)
dest_city = st.sidebar.selectbox("Ciudad de destino", dest_city_classes)
origin_state = st.sidebar.selectbox("Estado de origen", origin_state_classes)
dest_state = st.sidebar.selectbox("Estado de destino", dest_state_classes)

week_type = st.sidebar.selectbox("Tipo de semana", ["Laboral", "Fin de semana"])

crs_dep_time = st.sidebar.slider("Scheduled departure time (military format)", 0, 2359, 900)
crs_arr_time = st.sidebar.slider("Scheduled time of arrival (military format)", 0, 2359, 1130)
distance = st.sidebar.number_input("Distance (in miles)", 100, 5000, 2500)
quarter = st.sidebar.selectbox("Quarter", [1, 2, 3, 4])
month = st.sidebar.slider("Month", 1, 12, 6)
day_of_month = st.sidebar.slider("Day of the month", 1, 31, 15)

# Crear un DataFrame con los datos del usuario
input_data = pd.DataFrame({
    "Airline": [airline],
    "Origin": [origin],
    "Dest": [dest],
    "OriginCityName": [origin_city],
    "DestCityName": [dest_city],
    "OriginStateName": [origin_state],
    "DestStateName": [dest_state],
    "CRSDepTime": [crs_dep_time],
    "CRSArrTime": [crs_arr_time],
    "Distance": [distance],
    "Quarter": [quarter],
    "Month": [month],
    "DayofMonth": [day_of_month],
    "WeekType": [week_type]
})

st.write("### Datos de entrada")
st.dataframe(input_data)

# Preprocesar los datos antes de la predicción

try:
    transformed_input = {
        "Airline": airline_classes.get(input_data.get("Airline", "")), 
        "Origin": origin_classes.get(input_data.get("Origin", "")), 
        "Dest": dest_classes.get(input_data.get("Dest", "")), 
        "OriginCityName": origin_city_classes.get(input_data.get("OriginCityName", "")), 
        "DestCityName": dest_city_classes.get(input_data.get("DestCityName", "")), 
        "OriginStateName": origin_state_classes.get(input_data.get("OriginStateName", "")), 
        "DestStateName": dest_state_classes.get(input_data.get("DestStateName", "")), 
        "CRSDepTime": input_data.get("CRSDepTime", 0), 
        "CRSArrTime": input_data.get("CRSArrTime", 0), 
        "Distance": input_data.get("Distance", 0), 
        "Quarter": input_data.get("Quarter", 0), 
        "Month": input_data.get("Month", 0), 
        "DayofMonth": input_data.get("DayofMonth", 0), 
        "DayOfWeek": input_data.get("DayOfWeek", 0),
    }

    # Crea un DataFrame para la predicción
    df_transformed = pd.DataFrame([transformed_input])
    st.write("Datos transformados listos para predecir:")
    st.write(df_transformed)

except Exception as e:
    st.error(f"Hubo un error al transformar los datos: {e}")

# Predicción
if st.button("Predecir retraso"):
    prediction = modelo.predict(df_transformed)  # Asegúrate de que df_transformed esté definido correctamente

    if prediction == 0:
        st.write("✅ Afortunadamente, su vuelo no se ha retrasado.")
    else:
        st.write("⚠️ Desafortunadamente, su vuelo ha sido retrasado. Por favor, tome las medidas necesarias.")
