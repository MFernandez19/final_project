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


#Convertir hora en bloques de 15 minutos
def convert_military_to_quarter_hour(dep_time):
    return round(dep_time * 4) / 4  #Cuarto de hora mas cercano

#Interfaz de streamlit
st.title("Modelo de predicción de Retrasos en Vuelos ✈️")
st.write("Ingrese los detalles del vuelo para obtener una predicción de retraso.")

#Inputs del usuario
airline = st.selectbox("Selecciona la aerolínea", ["Airline"])
origin = st.selectbox("Aeropuerto de origen", ["OriginCityName"] )
dest = st.selectbox("Aeropuerto de destino", ["OriginCityName"] )
dep_time = st.slider("Hora de salida (formato 24h)", min_value = 0.0, max_value = 23.75, value = 12.0, step=0.25)  #Ajustado para permitir cuartos de hora
day = st.date_input("Fecha del vuelo", datetime.date.today())

#Calcular HourBlock
hour_block = convert_military_to_quarter_hour(dep_time)

#Boton para predecir
if st.button("Predecir Retraso"):
    input_data = pd.DataFrame({
        'DayOfWeek': [day.weekday() + 1],
        'Month': [day.month],
        'DayofMonth': [day.day],
        'HourBlock': [dep_time],  
        'Airline': [Airline],
        'Origin': [OriginCityName],
        'Dest': [OriginCityName.index(dest)],
    })
    
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
