#Cargar modelo
Model_path= "/workspace/final_project/models/flight_delay_model.pkl" 
with open("/workspace/final_project/models/flight_delay_model.pkl", "rb") as file:
    model = pickle.load(file)


Aerolineas = ['Airline'] 
Aeropuertos = ['OriginCityName']  

#Convertir hora en bloques de 15 minutos
def convert_military_to_quarter_hour(dep_time):
    return round(dep_time * 4) / 4  #Cuarto de hora mas cercano

#Interfaz de streamlit
st.title("Modelo de predicción de Retrasos en Vuelos ✈️")
st.write("Ingrese los detalles del vuelo para obtener una predicción de retraso.")

#Inputs del usuario
airline = st.selectbox("Selecciona la aerolínea", Aerolineas)
origin = st.selectbox("Aeropuerto de origen", Aeropuertos)
dest = st.selectbox("Aeropuerto de destino", Aeropuertos)
dep_time = st.slider("Hora de salida (formato 24h)", 0, 23.75, 12.0, step=0.25)  #Ajustado para permitir cuartos de hora
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
        'Airline': [Aerolineas.index(airline)],
        'Origin': [Aeropuertos.index(origin)],
        'Dest': [Aeropuertos.index(dest)],
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
