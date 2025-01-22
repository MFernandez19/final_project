# Basic imports
import pandas as pd
import numpy as np 

# Functional imports

import dump
import gc
import streamlit as st
import datetime
from pickle import load


#Cargar modelo

with open("../models/flight_delay_model.pkl", "rb") as f:
    model = load(f)

df_valores_unicos_cat = pd.read_csv("../data/raw/df_valores_unicos.csv")
df_valores_unicos_origin = pd.read_csv("../data/raw/df_valores_unicos_origin.csv")

#Convertir hora en bloques de 15 minutos
def convert_military_to_quarter_hour(dep_time):
    return round(dep_time * 4) / 4  #Cuarto de hora mas cercano

#Interfaz de streamlit
st.title("Modelo de predicción de Retrasos en Vuelos ✈️")
st.write("Ingrese los detalles del vuelo para obtener una predicción de retraso.")

#Inputs del usuario
Airline = st.selectbox("Selecciona la aerolínea", df_valores_unicos_cat["Airline"])
OriginCityName = st.selectbox("Aeropuerto de origen", df_valores_unicos_origin["OriginCityName"] )
OriginCityName = st.selectbox("Aeropuerto de destino", df_valores_unicos_origin["OriginCityName"] )
DepTime = st.slider("Hora de salida (formato 24h)", min_value = 0.0, max_value = 23.75, value = 12.0, step=0.25)  #Ajustado para permitir cuartos de hora
DayOfWeek = st.date_input("Fecha del vuelo", datetime.date.today())

#Calcular HourBlock
hour_block = convert_military_to_quarter_hour(DepTime)

#Boton para predecir
if st.button("Predecir Retraso"):
    input_data = df_input_data["Airline", "OriginCityName", "DayOfWeek", "DepTime"]

    #hacer prediccion
    prediction = model.predict(input_data)[0]
    
    #Formatear prediccion
    def format_delay_time(minutes):
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        if hours > 0:
            return f"{hours}h {mins}min"
        return f"{mins}min"
    
    delay_time = format_delay_time(max(0, prediction))  #No neg vals.
    
    #Resultado
    if prediction > 15:
        st.error(f"⚠️ Retraso estimado: {delay_time}")
    else:
        st.success(f"✅ El vuelo probablemente salga a tiempo ({delay_time} de retraso).")
