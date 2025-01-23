import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Título de la app
st.title("Predicción de retrasos en vuelos")

# Cargar modelo entrenado
@st.cache_resource
def load_model():
    with open("../models/flight_delay_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

df_valores_unicos_origin = pd.read_csv("../data/raw/df_valores_unicos_origin.csv")
df_valores_unicos_cat = pd.read_csv("../data/raw/df_valores_unicos.csv")

@st.cache_data
def load_data():
    # Simulación de datos
    data = pd.DataFrame({
        "Airline": df_valores_unicos_cat[0],
        "OriginCityName": df_valores_unicos_origin[0],
        "DepTime": [600, 1230, 945, 1500, 1730] * 20,
        "DayOfWeek": [1, 2, 3, 4, 5],
        "Delay": [15, 30, 5, 45, 60]  # Retraso en minutos
    })
    return data

data = load_data()

#Convertir hora en bloques de 15 minutos
def convert_military_to_quarter_hour(dep_time):
    return round(dep_time * 4) / 4  #Cuarto de hora mas cercano

st.title("Modelo de predicción de Retrasos en Vuelos ✈️")
st.write("Ingrese los detalles del vuelo para obtener una predicción de retraso.")

# Mostrar datos
st.write("### Datos de vuelos")
st.dataframe(data.head())

# Selección de características y variable objetivo
x = data[["Airline", "OriginCityName", "DepTime", "DayOfWeek"]]
y = data["Delay"]

# Preprocesamiento
label_encoders = {}

for col in ["Airline", "OriginCityName"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Entrada del usuario
st.write("### Realiza una Predicción")

airline = st.selectbox("Selecciona la aerolínea", label_encoders["Airline"].classes_)
origin_city = st.selectbox("Selecciona la ciudad de origen", label_encoders["OriginCityName"].classes_)
DepTime = st.slider("Hora de salida (en formato militar)", min_value=0, max_value=2359, step=1)
DayOfWeek = st.selectbox("Día de la semana", [1, 2, 3, 4, 5, 6, 7])

# Transformar los datos de entrada del usuario
input_data = pd.DataFrame({
    "Airline": [label_encoders["Airline"].transform([airline])[0]],
    "OriginCityName": [label_encoders["OriginCityName"].transform([origin_city])[0]],
    "DepTime": [DepTime],
    "DayOfWeek": [DayOfWeek]
})

# Realizar la predicción
if st.button("Predecir retraso"):
    prediction = model.predict(input_data)[0]
    st.write(f"### Retraso estimado: {prediction:.2f} minutos")