import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Título de la aplicación
st.title("Predicción de Retrasos en Vuelos")

# Cargar el modelo preentrenado
@st.cache_resource
def load_model():
    with open("/workspace/final_project/models/best_model_xgb_subsample_1.0_n_estimators_200_max_depth_10_learning_rate_0.2_gamma_0.1_colsample_bytree_0.8.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Cargar las clases de codificación (LabelEncoder)
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

# Crear los codificadores (LabelEncoders) para cada columna
encoders = {
    "Airline": LabelEncoder().fit(list(airline_classes.values())),
    "Origin": LabelEncoder().fit(list(origin_classes.values())),
    "Dest": LabelEncoder().fit(list(dest_classes.values())),
    "OriginCityName": LabelEncoder().fit(list(origin_city_classes.values())),
    "DestCityName": LabelEncoder().fit(list(dest_city_classes.values())),
    "OriginStateName": LabelEncoder().fit(list(origin_state_classes.values())),
    "DestStateName": LabelEncoder().fit(list(dest_state_classes.values()))
}

# Usar un scaler para las columnas numéricas
scaler = StandardScaler()

# Columnas utilizadas en el modelo
features = [
    "Airline", "Origin", "Dest", "OriginCityName", "DestCityName", 
    "OriginStateName", "DestStateName", "CRSDepTime", "CRSArrTime",
    "Distance", "Quarter", "Month", "DayofMonth", "DayOfWeek", "dia_festivo"
]

# Función para preprocesar los datos
@st.cache_resource
def preprocess_data(input_data, encoders, scaler):
    # Codificar características categóricas
    for col, encoder in encoders.items():
        input_data[col] = encoder.transform(input_data[col])
    
    # Escalar características numéricas
    numeric_cols = ["CRSDepTime", "CRSArrTime", "Distance", "Quarter", "Month", "DayofMonth", "DayOfWeek"]
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    
    return input_data

# Entradas del usuario
st.sidebar.header("Introducir características del vuelo")

airline = st.sidebar.selectbox("Aerolínea", airline_classes)
origin = st.sidebar.selectbox("Código de aeropuerto de origen", list(origin_classes.keys()))
dest = st.sidebar.selectbox("Código de aeropuerto de destino", list(dest_classes.keys()))
origin_city = st.sidebar.selectbox("Ciudad de origen", origin_city_classes)
dest_city = st.sidebar.selectbox("Ciudad de destino", dest_city_classes)
origin_state = st.sidebar.selectbox("Estado de origen", origin_state_classes)
dest_state = st.sidebar.selectbox("Estado de destino", dest_state_classes)

crs_dep_time = st.sidebar.slider("Scheduled departure time (military format)", 0, 2359, 900)
crs_arr_time = st.sidebar.slider("Scheduled time of arrival (military format)", 0, 2359, 1130)
distance = st.sidebar.number_input("Distance (in miles)", 100, 5000, 2500)
quarter = st.sidebar.selectbox("Quarter", [1, 2, 3, 4])
month = st.sidebar.slider("Month", 1, 12, 6)
day_of_month = st.sidebar.slider("Day of the month", 1, 31, 15)

# Calcular el día de la semana basado en la fecha ingresada (suponiendo año 2025)
day_of_week = datetime.datetime(2025, month, day_of_month).weekday() + 1  # Ajustar a 1-7

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
    "DayOfWeek": [day_of_week]
})

st.write("### Datos de entrada")
st.dataframe(input_data)

# Preprocesar los datos antes de la predicción
try:
    # Transforma las clases del usuario asegurando que los valores son escalares
    transformed_input = {
        "Airline": airline_classes.get(input_data["Airline"].iloc[0], ""),
        "Origin": origin_classes.get(input_data["Origin"].iloc[0], ""),
        "Dest": dest_classes.get(input_data["Dest"].iloc[0], ""),
        "OriginCityName": origin_city_classes.get(input_data["OriginCityName"].iloc[0], ""),
        "DestCityName": dest_city_classes.get(input_data["DestCityName"].iloc[0], ""),
        "OriginStateName": origin_state_classes.get(input_data["OriginStateName"].iloc[0], ""),
        "DestStateName": dest_state_classes.get(input_data["DestStateName"].iloc[0], ""),
        "CRSDepTime": input_data["CRSDepTime"].iloc[0],
        "CRSArrTime": input_data["CRSArrTime"].iloc[0],
        "Distance": input_data["Distance"].iloc[0],
        "Quarter": input_data["Quarter"].iloc[0],
        "Month": input_data["Month"].iloc[0],
        "DayofMonth": input_data["DayofMonth"].iloc[0],
        "DayOfWeek": input_data["DayOfWeek"].iloc[0]
    }

    # Crea un DataFrame para la predicción
    df_transformed = pd.DataFrame([transformed_input])
    st.write("Datos transformados listos para predecir:")
    st.write(df_transformed)

except Exception as e:
    st.error(f"Hubo un error al transformar los datos: {e}")

# Predicción
if st.button("Predecir retraso"):
    prediction = model.predict(df_transformed)  

    if prediction == 0:
        st.write("✅ Afortunadamente, su vuelo no se ha retrasado.")
    else:
        st.write("⚠️ Desafortunadamente, su vuelo ha sido retrasado. Por favor, tome las medidas necesarias.")
