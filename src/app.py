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
    with open("/workspace/final_project/models/best_model_xgb_subsample_1.0_n_estimators_200_max_depth_10_learning_rate_0.2_gamma_0.1_colsample_bytree_0.8.pkl", "rb") as file:
        model = pickle.load(file)
    return model

df_datos = pd.read_csv("/data/processed/X_test_with_outliers_norm.csv")

model = load_model()

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
airlines_data = df_datos["Airline"]
origin_state_data = df_datos["OriginStateName"]
dest_state_data = df_datos["DestStateName"]
origin_city_data = df_datos["OriginCityName"]
dest_city_data = df_datos["DestCityName"]
airports_data = ["LAX", "JFK", "ORD", "IAH", "MIA"]
days_of_week_data = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

airline = st.sidebar.selectbox("Aerolínea", airlines_data)
origin = st.sidebar.text_input("Código de aeropuerto de origen", "JFK")
dest = st.sidebar.text_input("Código de aeropuerto de destino", "JFK")
origin_city = st.sidebar.selectbox("Ciudad de origen", cities)
dest_city = st.sidebar.selectbox("Ciudad de destino", cities)
origin_state = st.sidebar.selectbox("Estado de origen", origin_state_data)
dest_state = st.sidebar.selectbox("Estado de destino", dest_state_data)

day_of_week = st.sidebar.selectbox("Weekday", days_of_week_data)
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
        "Airline": LabelEncoder().fit(airlines),
        "Origin": LabelEncoder().fit(airports),
        "Dest": LabelEncoder().fit(airports),
        "OriginCityName": LabelEncoder().fit(cities),
        "DestCityName": LabelEncoder().fit(cities),
        "OriginStateName": LabelEncoder().fit(states),
        "DestStateName": LabelEncoder().fit(states),
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