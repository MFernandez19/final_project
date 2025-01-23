import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Título de la aplicación
st.title("Predicción de Retrasos en Vuelos")

# Cargar el modelo preentrenado
@st.cache_resource
def load_model():
    with open("../models/best_model_xgb_subsample_1.0_n_estimators_200_max_depth_10_learning_rate_0.2_gamma_0.1_colsample_bytree_0.8.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Columnas utilizadas en el modelo
features = [
    "Airline", "Origin", "Dest", "OriginCityName", "DestCityName",
    "OriginStateName", "DestStateName", "CRSDepTime", "CRSArrTime",
    "Distance", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "WeekType", "dia_festivo"
]

# Función para preprocesar los datos
@st.cache_data
def preprocess_data(input_data, encoders, scaler):
    # Codificar características categóricas
    for col in encoders:
        if col in input_data:
            input_data[col] = encoders[col].transform(input_data[col])
    
    # Escalar características numéricas
    numeric_cols = ["CRSDepTime", "CRSArrTime", "Distance", "Quarter", "Month", "DayofMonth", "DayOfWeek"]
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    
    return input_data

# Entradas del usuario
st.sidebar.header("Introducir características del vuelo")
airline = st.sidebar.selectbox("Aerolínea", ["AirlineA", "AirlineB", "AirlineC", "AirlineD"])
origin = st.sidebar.text_input("Código de aeropuerto de origen", "JFK")
dest = st.sidebar.text_input("Código de aeropuerto de destino", "LAX")
origin_city = st.sidebar.text_input("Ciudad de origen", "New York")
dest_city = st.sidebar.text_input("Ciudad de destino", "Los Angeles")
origin_state = st.sidebar.selectbox("Estado de origen", ["New York", "California", "Texas", "Florida"])
dest_state = st.sidebar.selectbox("Estado de destino", ["New York", "California", "Texas", "Florida"])
crs_dep_time = st.sidebar.slider("Hora de salida programada (formato militar)", 0, 2359, 900)
crs_arr_time = st.sidebar.slider("Hora de llegada programada (formato militar)", 0, 2359, 1130)
distance = st.sidebar.number_input("Distancia (en millas)", 100, 5000, 2500)
quarter = st.sidebar.selectbox("Trimestre", [1, 2, 3, 4])
month = st.sidebar.slider("Mes", 1, 12, 6)
day_of_month = st.sidebar.slider("Día del mes", 1, 31, 15)
day_of_week = st.sidebar.selectbox("Día de la semana", [0, 1, 2, 3, 4, 5, 6])  # Lunes = 0, Domingo = 6
week_type = st.sidebar.selectbox("Tipo de semana", ["Laboral", "Fin de semana"])
dia_festivo = st.sidebar.selectbox("¿Es día festivo?", ["Sí", "No"])

# Crear un DataFrame con los datos del usuario
try:
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
except ValueError as e:
    st.error(f"Error al crear el DataFrame: {e}")

st.write("### Datos de entrada")
st.dataframe(input_data)

# Preprocesar los datos antes de la predicción
@st.cache_resource
def load_encoders_and_scaler():
    encoders = {
        "Airline": LabelEncoder().fit(["AirlineA", "AirlineB", "AirlineC", "AirlineD"]),
        "Origin": LabelEncoder().fit(["JFK", "LAX", "ORD", "ATL"]),
        "Dest": LabelEncoder().fit(["JFK", "LAX", "ORD", "ATL"]),
        "OriginCityName": LabelEncoder().fit(["New York", "Los Angeles", "Chicago", "Atlanta"]),
        "DestCityName": LabelEncoder().fit(["New York", "Los Angeles", "Chicago", "Atlanta"]),
        "OriginStateName": LabelEncoder().fit(["New York", "California", "Texas", "Florida"]),
        "DestStateName": LabelEncoder().fit(["New York", "California", "Texas", "Florida"]),
        "WeekType": LabelEncoder().fit(["Laboral", "Fin de semana"]),
    }
    scaler = StandardScaler().fit(pd.DataFrame({
        "CRSDepTime": [0, 2359],
        "CRSArrTime": [0, 2359],
        "Distance": [0, 5000],
        "Quarter": [1, 4],
        "Month": [1, 12],
        "DayofMonth": [1, 31],
        "DayOfWeek": [1, 7],
        "WeekType": [0, 1],
        "dia_festivo": [0, 1]
    }))

    return encoders, scaler

encoders, scaler = load_encoders_and_scaler()

# Preprocesar los datos
processed_data = preprocess_data(input_data, _encoders, scaler)

# Predicción
if st.button("Predecir retraso"):
    prediction = model.predict(processed_data)[0]
    st.write(f"### Predicción del retraso: {prediction:.2f} minutos")
