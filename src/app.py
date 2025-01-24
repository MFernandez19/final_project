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

# Columnas utilizadas en el modelo
features = [
    "Airline", "Origin", "Dest", "OriginCityName", "DestCityName", "OriginStateName", "DestStateName", "CRSDepTime", "CRSArrTime",
    "Distance", "Quarter", "Month", "DayofMonth", "DayOfWeek", "WeekType", "dia_festivo"
]

# Función para preprocesar los datos
@st.cache_resource
def preprocess_data(input_data, _encoders, _scaler):
    # Codificar características categóricas
    for col, encoder in _encoders.items():
        input_data[col] = encoder.transform(input_data[col])
    
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
    transformed_input = input_data.copy()
    
    # Transformar las clases de texto a números
    transformed_input["Airline"] = airline_classes[airline]
    transformed_input["Origin"] = origin_classes.get(origin, "")
    transformed_input["Dest"] = dest_classes.get(dest, "")
    transformed_input["OriginCityName"] = origin_city_classes.get(origin_city, "")
    transformed_input["DestCityName"] = dest_city_classes.get(dest_city, "")
    transformed_input["OriginStateName"] = origin_state_classes.get(origin_state, "")
    transformed_input["DestStateName"] = dest_state_classes.get(dest_state, "")
    
    # Asegurarse de que los números están bien pasados (puede que tu modelo no maneje mal valores no numéricos)
    transformed_input["CRSDepTime"] = int(crs_dep_time)
    transformed_input["CRSArrTime"] = int(crs_arr_time)
    transformed_input["Distance"] = int(distance)
    transformed_input["Quarter"] = int(quarter)
    transformed_input["Month"] = int(month)
    transformed_input["DayofMonth"] = int(day_of_month)
    
    # Aquí va el preprocesamiento final con los encoders y el escalado
    df_transformed = preprocess_data(transformed_input, encoders, scaler)
    
except Exception as e:
    st.error(f"Hubo un error al transformar los datos: {e}")

# Predicción
try:
    prediction = model.predict(df_transformed)
    if prediction[0] == 0:
        st.success("Afortunadamente su vuelo no se ha retrasado.")
    else:
        st.error("Desafortunadamente su vuelo ha sido retrasado. Por favor, tome las precauciones necesarias.")
except Exception as e:
    st.error(f"Hubo un error al realizar la predicción: {e}")
