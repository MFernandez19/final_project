# Proyecto Final: Predicción de Retrasos en Vuelos

## 🌍 Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un sistema predictivo que permita anticipar retrasos en vuelos comerciales utilizando datos históricos de vuelos en los Estados Unidos. La predicción de retrasos es un problema recurrente en la industria aérea que impacta significativamente a aerolíneas, aeropuertos y pasajeros.

Para abordar este problema, utilizamos datos del Flight Delay Dataset 2018-2022, disponible en Kaggle. Sin embargo, debido a limitaciones computacionales, restringimos nuestro análisis y modelado exclusivamente a los vuelos del año 2021.

El modelo final está basado en XGBoost con optimización de hiperparámetros mediante RandomizedSearchCV, logrando una precisión del 84.52% en la clasificación binaria de vuelos retrasados.

## 🌟 Objetivos del Proyecto

Este sistema predictivo busca:

✅ Ayudar a aerolíneas y aeropuertos a identificar patrones operacionales que generan retrasos.

✅ Optimizar la planificación de vuelos mediante la detección temprana de riesgos de demoras.

✅ Mejorar la experiencia del pasajero, permitiendo prever retrasos y tomar mejores decisiones de viaje.

## 📊 Fuente de Datos

Fuente: Kaggle (Flight Delay Dataset 2018-2022)

Período Analizado: Solo 2021

Formato: .parquet

Variables Clave:

Tiempo de Salida y Tiempo de Llegada

Aerolínea

Aeropuerto de Origen/Destino

Retrasos Previos

Condiciones Meteorológicas

## 🔄 Preprocesamiento y Análisis Exploratorio (EDA)

Durante el EDA se identificaron los siguientes puntos clave:

🌐 Distribución de retrasos: La mayoría de los retrasos fueron menores a 30 minutos.

📊 Factores determinantes: Los retrasos están influenciados por la aerolínea, aeropuerto y horario del vuelo.

⚙️ Preprocesamiento:

Transformación de datos categóricos.

Normalización con StandardScaler y MinMaxScaler.

Manejo de desequilibrio en los datos con SMOTE y RandomUnderSampler.

Visualizaciones con calmap para analizar patrones temporales de retraso.

## 🔧 Estructura del Proyecto

FINAL_PROJECT/

├── .devcontainer/

├── .vscode/

├── data/

│   ├── interim/

│   ├── processed/ (.gitkeep)

│   └── raw/ (.gitkeep)

├── models/ (.gitkeep)

├── src/

│   └── pruebas/

│       ├── explore.ipynb

│       ├── prueba.ipynb

│       └── ultima_prueba.ipynb
├── app.py

├── best_model_xgb_subsample_1.0_n_esti...

├── Combined_Flights_2021_streamlit.parquet

├── EDA.ipynb

├── render.txt

├── requirements.txt

├── .env.example

├── .gitignore

├── .gitpod.yml

├── README.es.md

└── README.md

## 🤖 Modelo de Predicción

Modelo Base: XGBoost Clasificación Binaria

Accuracy: 83.80%

Ajuste de hiperparámetros: RandomizedSearchCV

Se optimizó para reducir falsos negativos (FN)

Modelo Final: XGBoost + RandomSearchCV

Accuracy: 84.52%

Reducción de FN y optimización de recall

## 🚨 Limitaciones del Modelo

📂 Almacenamiento: Se requiere optimización de recursos para manejar el volumen de datos.

🌐 Implementación en producción: Integración con plataformas en tiempo real.

💻 Diseño del modelo: Posibles mejoras con redes neuronales o modelos secuenciales.

## 🛠️ Configuración e Instalación

🔗 Prerrequisitos

Asegúrate de tener instalado:

Python 3.8+

pip (gestor de paquetes de Python)

🔗 Instalación

1. Clona este repositorio:

git clone https://github.com/tuusuario/Flight-Status-Prediction.git
cd Flight-Status-Prediction

2. Instala las dependencias:

pip install -r requirements.txt

## ⚡ Ejecutando la Aplicación

Ejecuta el script principal:

python app.py

Si deseas re-entrenar el modelo:

python app.py --train

## 🎉 Conclusión

Este sistema proporciona una herramienta predictiva útil para aerolíneas y pasajeros, optimizando la toma de decisiones y reduciendo el impacto de los retrasos en los vuelos.

🚀 ¡Gracias por visitar nuestro proyecto!

