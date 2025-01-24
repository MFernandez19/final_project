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

with open("data/interim/enc_Airline.json", "r") as file:
    content1 = json.load(file)

print(content1)

# Carga manual
with open("data/interim/enc_DestCityName.json", "r") as file:
    content2 = json.load(file)

print(content2)

# Carga manual
with open("data/interim/enc_DestStateName.json", "r") as file:
    content3 = json.load(file)

print(content3)

# Carga manual
with open("data/interim/enc_OriginCityName.json", "r") as file:
    content4 = json.load(file)

print(content4)

# Carga manual
with open("data/interim/enc_OriginStateName.json", "r") as file:
    content5 = json.load(file)

print(content5)

# Carga manual
with open("data/interim/enc_Origin.json", "r") as file:
    content6 = json.load(file)

print(content6)

# Carga manual
with open("data/interim/enc_Dest.json", "r") as file:
    content7 = json.load(file)

print(content7)

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
airlines = df_datos_airline
states = ["California", "Texas", "Florida", "New York", "Illinois"]
cities = ["Los Angeles", "New York", "Chicago", "Houston", "Miami"]
airports = ["LAX", "JFK", "ORD", "IAH", "MIA"]
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

airline = st.sidebar.selectbox("Aerolínea", content1)
origin = st.sidebar.text_input("Código de aeropuerto de origen", content6)
dest = st.sidebar.text_input("Código de aeropuerto de destino", content7)
origin_city = st.sidebar.selectbox("Ciudad de origen", content4)
dest_city = st.sidebar.selectbox("Ciudad de destino", content2)
origin_state = st.sidebar.selectbox("Estado de origen", content5)
dest_state = st.sidebar.selectbox("Estado de destino", content3)

day_of_week = st.sidebar.selectbox("Weekday", days_of_week)
week_type = st.sidebar.selectbox("Tipo de semana", ["Laboral", "Fin de semana"])
dia_festivo = st.sidebar.selectbox("¿Es día festivo?", ["Sí", "No"])

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
    "DayOfWeek": [day_of_week],
    "WeekType": [week_type],
    "dia_festivo": [1 if dia_festivo == "Sí" else 0]
})

st.write("### Datos de entrada")
st.dataframe(input_data)

# Preprocesar los datos antes de la predicción
@st.cache_resource
def load_encoders_and_scaler():
    encoders = {
        "Airline": LabelEncoder().fit(df_datos_airline),
        "Origin": LabelEncoder().fit(df_datos_Origin),
        "Dest": LabelEncoder().fit(df_datos_Dest),
        "OriginCityName": LabelEncoder().fit(df_datos_OriginCityName),
        "DestCityName": LabelEncoder().fit(df_datos_DestCityName),
        "OriginStateName": LabelEncoder().fit(df_datos_OriginStateName),
        "DestStateName": LabelEncoder().fit(df_datos_DestStateName),
        "WeekType": LabelEncoder().fit(["Laboral", "Fin de semana"]),
        "DayOfWeek": LabelEncoder().fit(days_of_week)
    }
    scaler = StandardScaler().fit(pd.DataFrame({
        "CRSDepTime": [0, 1200, 2359, 1800],
        "CRSArrTime": [0, 1200, 2359, 1800],
        "Distance": [100, 2500, 5000, 1500],
        "Quarter": [1, 2, 3, 4],
        "Month": [1, 6, 12, 3],
        "DayofMonth": [1, 15, 31, 10],
        "DayOfWeek": [0, 1, 2, 3]
    }))  # Asegurar que todas las listas tengan la misma longitud
    return encoders, scaler

encoders, scaler = load_encoders_and_scaler()

# Preprocesar los datos
processed_data = preprocess_data(input_data, encoders, scaler)

# Asegurar que las columnas coincidan con las del modelo
processed_data = processed_data.reindex(columns=features, fill_value=0)

# Predicción
if st.button("Predecir retraso"):
    prediction = model.predict(processed_data)[0]
    st.write(f"### Predicción del retraso: {prediction:.2f} minutos")