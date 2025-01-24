import pandas as pd
import streamlit as st
import datetime
import pickle
import json

# Ruta al archivo del DataFrame y al modelo
FILE_PATH = "../data/raw/Combined_Flights_2021.csv"
MODEL_PATH = "../models/best_model_xgb_subsample_1.0_n_estimators_200_max_depth_10_learning_rate_0.2_gamma_0.1_colsample_bytree_0.8.pkl"

required_columns = [
    "Airline", "Origin", "Dest", "OriginCityName", "DestCityName",
    "OriginStateName", "DestStateName"
]

# Cargar el DataFrame
@st.cache_resource
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df[required_columns].drop_duplicates().dropna()
    return df

# Cargar los encoders
@st.cache_resource
def load_encoders():
    encoders = {}
    files = [
        "enc_Airline", "enc_Origin", "enc_Dest", 
        "enc_OriginCityName", "enc_DestCityName", 
        "enc_OriginStateName", "enc_DestStateName"
    ]
    for file in files:
        with open(f"../data/interim/{file}.json", "r") as f:
            encoders[file.split("_")[1]] = json.load(f)
    return encoders

# Cargar el modelo
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

# Crear mapeos dinámicos
@st.cache_resource
def create_mappings(data):
    state_to_city_mapping = data.groupby("OriginStateName")["OriginCityName"].apply(lambda x: list(set(x))).to_dict()
    city_to_airport_mapping = data.groupby("OriginCityName")["Origin"].apply(lambda x: list(set(x))).to_dict()
    dest_state_to_city_mapping = data.groupby("DestStateName")["DestCityName"].apply(lambda x: list(set(x))).to_dict()
    dest_city_to_airport_mapping = data.groupby("DestCityName")["Dest"].apply(lambda x: list(set(x))).to_dict()
    return state_to_city_mapping, city_to_airport_mapping, dest_state_to_city_mapping, dest_city_to_airport_mapping

# Preparar los datos
data = load_and_prepare_data(FILE_PATH)
encoders = load_encoders()
model = load_model()
state_to_city_mapping, city_to_airport_mapping, dest_state_to_city_mapping, dest_city_to_airport_mapping = create_mappings(data)

# Aplicación Streamlit
st.title("Predicción de Retrasos en Vuelos")

# Barra lateral para entrada del usuario
st.sidebar.header("Introducir características del vuelo")

# Selección dinámica para el origen
origin_state = st.sidebar.selectbox("Estado de origen", list(state_to_city_mapping.keys()))
origin_city_options = state_to_city_mapping[origin_state]
origin_city = st.sidebar.selectbox("Ciudad de origen", origin_city_options)

origin_airport_options = city_to_airport_mapping[origin_city]
origin_airport = st.sidebar.selectbox("Aeropuerto de origen", origin_airport_options)

# Selección dinámica para el destino
dest_state = st.sidebar.selectbox("Estado de destino", list(dest_state_to_city_mapping.keys()))
dest_city_options = dest_state_to_city_mapping[dest_state]
dest_city = st.sidebar.selectbox("Ciudad de destino", dest_city_options)

dest_airport_options = dest_city_to_airport_mapping[dest_city]
dest_airport = st.sidebar.selectbox("Aeropuerto de destino", dest_airport_options)

# Otras entradas del usuario
airline = st.sidebar.selectbox("Aerolínea", data["Airline"].unique())
departure_time = st.sidebar.time_input("Hora de salida (24h)", value=datetime.time(9, 0))
arrival_time = st.sidebar.time_input("Hora de llegada (24h)", value=datetime.time(16, 30))
distance = st.sidebar.number_input("Distancia (en millas)", 100, 5000, 2500)
flight_date = st.sidebar.date_input("Fecha del vuelo", min_value=datetime.date(2023, 1, 1))

# Convertir la fecha y hora a los datos necesarios
day_of_month = flight_date.day
month = flight_date.month
day_of_week = flight_date.weekday() + 1  # Lunes=1, Domingo=7
quarter = (month - 1) // 3 + 1

# Convertir la hora a formato militar (HHMM)
crs_dep_time = departure_time.hour * 100 + departure_time.minute
crs_arr_time = arrival_time.hour * 100 + arrival_time.minute

# Transformar las variables categóricas con los encoders
try:
    airline_encoded = encoders["Airline"].get(airline, -1)
    origin_encoded = encoders["Origin"].get(origin_airport, -1)
    dest_encoded = encoders["Dest"].get(dest_airport, -1)
    origin_city_encoded = encoders["OriginCityName"].get(origin_city, -1)
    dest_city_encoded = encoders["DestCityName"].get(dest_city, -1)
    origin_state_encoded = encoders["OriginStateName"].get(origin_state, -1)
    dest_state_encoded = encoders["DestStateName"].get(dest_state, -1)
except KeyError as e:
    st.error(f"Error al codificar las variables: {e}")

# Crear un diccionario con las entradas codificadas
input_data = {
    "Airline": airline_encoded,
    "Origin": origin_encoded,
    "Dest": dest_encoded,
    "OriginCityName": origin_city_encoded,
    "DestCityName": dest_city_encoded,
    "OriginStateName": origin_state_encoded,
    "DestStateName": dest_state_encoded,
    "CRSDepTime": crs_dep_time,
    "CRSArrTime": crs_arr_time,
    "Distance": distance,
    "Quarter": quarter,
    "Month": month,
    "DayofMonth": day_of_month,  # Corregido a "DayofMonth"
    "DayOfWeek": day_of_week,
}

# Mostrar los datos ingresados
st.write("### Datos de entrada codificados")
st.json(input_data)

# Realizar la predicción
if st.button("Predecir retraso"):
    try:
        # Crear DataFrame para el modelo
        df_transformed = pd.DataFrame([input_data])

        # Realizar la predicción
        prediction = model.predict(df_transformed)
        if prediction[0] == 0:
            st.success("✅ Afortunadamente, su vuelo no se ha retrasado.")
        else:
            st.warning("⚠️ Desafortunadamente, su vuelo ha sido retrasado. Por favor, tome las medidas necesarias.")
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")