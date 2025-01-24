import streamlit as st
import pandas as pd
import numpy as np
import pickle
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

# Cargar los diccionarios JSON para transformar datos categóricos
@st.cache_resource
def load_encoders():
    with open("/workspace/final_project/data/interim/enc_Airline.json", "r") as f:
        airline_classes = json.load(f)
    with open("/workspace/final_project/data/interim/enc_Dest.json", "r") as f:
        dest_classes = json.load(f)
    with open("/workspace/final_project/data/interim/enc_DestCityName.json", "r") as f:
        dest_city_classes = json.load(f)
    with open("/workspace/final_project/data/interim/enc_DestStateName.json", "r") as f:
        dest_state_classes = json.load(f)
    with open("/workspace/final_project/data/interim/enc_Origin.json", "r") as f:
        origin_classes = json.load(f)
    with open("/workspace/final_project/data/interim/enc_OriginCityName.json", "r") as f:
        origin_city_classes = json.load(f)
    with open("/workspace/final_project/data/interim/enc_OriginStateName.json", "r") as f:
        origin_state_classes = json.load(f)
    
    return {
        "Airline": airline_classes,
        "Origin": origin_classes,
        "Dest": dest_classes,
        "OriginCityName": origin_city_classes,
        "DestCityName": dest_city_classes,
        "OriginStateName": origin_state_classes,
        "DestStateName": dest_state_classes
    }

encoders = load_encoders()

# Columnas utilizadas en el modelo
features = [
    "Airline", "Origin", "Dest", "OriginCityName", "DestCityName", "OriginStateName", "DestStateName",
    "CRSDepTime", "CRSArrTime", "Distance", "Quarter", "Month", "DayofMonth", "DayOfWeek"
]

# Entradas del usuario
st.sidebar.header("Introducir características del vuelo")
user_inputs = {
    "Airline": st.sidebar.selectbox("Aerolínea", list(encoders["Airline"].keys())),
    "Origin": st.sidebar.selectbox("Código de aeropuerto de origen", list(encoders["Origin"].keys())),
    "Dest": st.sidebar.selectbox("Código de aeropuerto de destino", list(encoders["Dest"].keys())),
    "OriginCityName": st.sidebar.selectbox("Ciudad de origen", list(encoders["OriginCityName"].keys())),
    "DestCityName": st.sidebar.selectbox("Ciudad de destino", list(encoders["DestCityName"].keys())),
    "OriginStateName": st.sidebar.selectbox("Estado de origen", list(encoders["OriginStateName"].keys())),
    "DestStateName": st.sidebar.selectbox("Estado de destino", list(encoders["DestStateName"].keys())),
}

dia_festivo = st.sidebar.selectbox("¿Es día festivo?", ["Sí", "No"])
crs_dep_time = st.sidebar.slider("Scheduled departure time (military format)", 0, 2359, 900)
crs_arr_time = st.sidebar.slider("Scheduled time of arrival (military format)", 0, 2359, 1130)
distance = st.sidebar.number_input("Distance (in miles)", 100, 5000, 2500)
quarter = st.sidebar.selectbox("Quarter", [1, 2, 3, 4])
month = st.sidebar.slider("Month", 1, 12, 6)
day_of_month = st.sidebar.slider("Day of the month", 1, 31, 15)
day_of_week = st.sidebar.selectbox("Día de la semana", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# Transformar los datos de entrada usando los JSON cargados
transformed_input = {key: encoders[key][user_inputs[key]] for key in user_inputs}

# Agregar valores numéricos directamente
transformed_input.update({
    "CRSDepTime": crs_dep_time,
    "CRSArrTime": crs_arr_time,
    "Distance": distance,
    "Quarter": quarter,
    "Month": month,
    "DayofMonth": day_of_month,
    "DayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
})

input_data = pd.DataFrame([transformed_input])

st.write("### Datos de entrada")
st.dataframe(input_data)

# Asegurar que las columnas coincidan con las del modelo
processed_data = input_data.reindex(columns=features, fill_value=0)

# Verificación de los valores transformados antes de la predicción
st.write("### Datos transformados antes de la predicción")
st.dataframe(processed_data)

# Prueba de predicción con diferentes valores para verificar posibles sesgos
sample_predictions = model.predict(processed_data)
prediction_distribution = np.unique(sample_predictions, return_counts=True)

st.write("### Distribución de predicciones en el modelo")
st.write({"Valores Únicos": prediction_distribution[0], "Frecuencia": prediction_distribution[1]})

# Predicción
if st.button("Predecir retraso"):
    prediction = sample_predictions[0]
    if prediction == 0:
        st.success("Afortunadamente, su vuelo no tiene retraso previsto.")
    else:
        st.warning(f"Su vuelo podría tener un retraso de aproximadamente {prediction:.2f} minutos.")
