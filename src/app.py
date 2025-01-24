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

df_transformed = None  # Define df_transformed inicialmente como None

try:
    # Transforma las clases del usuario asegurando que los valores sean escalares
    transformed_input = {
        "Airline": airline_classes.get(input_data["Airline"], ""),
        "Origin": origin_classes.get(input_data["Origin"], ""),
        "Dest": dest_classes.get(input_data["Dest"], ""),
        "OriginCityName": origin_city_classes.get(input_data["OriginCityName"], ""),
        "DestCityName": dest_city_classes.get(input_data["DestCityName"], ""),
        "OriginStateName": origin_state_classes.get(input_data["OriginStateName"], ""),
        "DestStateName": dest_state_classes.get(input_data["DestStateName"], ""),
        "CRSDepTime": input_data["CRSDepTime"] if isinstance(input_data["CRSDepTime"], (int, float)) else input_data["CRSDepTime"].iloc[0],
        "CRSArrTime": input_data["CRSArrTime"] if isinstance(input_data["CRSArrTime"], (int, float)) else input_data["CRSArrTime"].iloc[0],
        "Distance": input_data["Distance"] if isinstance(input_data["Distance"], (int, float)) else input_data["Distance"].iloc[0],
        "Quarter": input_data["Quarter"] if isinstance(input_data["Quarter"], (int, float)) else input_data["Quarter"].iloc[0],
        "Month": input_data["Month"] if isinstance(input_data["Month"], (int, float)) else input_data["Month"].iloc[0],
        "DayofMonth": input_data["DayofMonth"] if isinstance(input_data["DayofMonth"], (int, float)) else input_data["DayofMonth"].iloc[0],
        "DayOfWeek": input_data["DayOfWeek"] if isinstance(input_data["DayOfWeek"], (int, float)) else input_data["DayOfWeek"].iloc[0],
    }

    # Crea un DataFrame con los datos transformados
    df_transformed = pd.DataFrame([transformed_input])

except Exception as e:
    st.error(f"Hubo un error al transformar los datos: {e}")

try:
    # Realiza la predicción si df_transformed es válido
    prediction = model.predict(df_transformed)
    
    if prediction[0] == 0:
        st.success("Afortunadamente su vuelo no se ha retrasado.")
    else:
        st.error("Desafortunadamente su vuelo ha sido retrasado. Por favor, tome las precauciones necesarias.")
except Exception as e:
    st.error(f"Hubo un error al realizar la predicción: {e}")


