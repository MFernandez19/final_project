# Proyecto Final Flight Status

El objetivo de este proyecto es desarrollar un sistema predictivo que permita identificar retrasos en vuelos comerciales basándonos en datos históricos. Usando el dataset proporcionado por Kaggle, que abarca información de vuelos desde 2018 hasta 2022, aquí vamos a analizar factores como aerolínea, aeropuerto, condiciones operativas y horarios para predecir si un vuelo tendrá un retraso significativo.

## Proposito

Los retrasos de vuelo son un problema recurrente en la industria aerea que afecta aerolíneas y pasajeros.
Este proyecto busca:

- Ayudar a aerolineas/aeropuertos a identificar patrones operativos que conducen a retrasos.
- Proporcionar herramientas predictivas para optimizar planificacion de vuelos.
- Mejorar la experiencia de los pasajeros para identificar posibles retrasos.

## Datos

El dataset usado en este proyecto proviene de kaggle (Flight Delay Dataset 2018-2022) que contiene información detallada sobre vuelos, aerolíneas, aeropuertos y factores asociados a retrasos.
Link: https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022/data

## Estructura

El proyecto está organizado de la siguiente manera:

- `app.py` - El script principal de Python que ejecutas para tu proyecto.
- `explore.py` - Un notebook para que puedas hacer tus exploraciones, idealmente el codigo de este notebook se migra hacia app.py para subir a produccion.
- `utils.py` - Este archivo contiene código de utilidad para operaciones como conexiones de base de datos.
- `requirements.txt` - Este archivo contiene la lista de paquetes de Python necesarios.
- `models/` - Este directorio debería contener tus clases de modelos SQLAlchemy.
- `data/` - Este directorio contiene los siguientes subdirectorios:
  - `interim/` - Para datos intermedios que han sido transformados.
  - `processed/` - Para los datos finales a utilizar para el modelado.
  - `raw/` - Aqui deben ir los archivos originales en formato .parquet.

## Nota sobre los datos 

Debido al considerable tamaño de el dataset, los archivos originales estan en formato .parquet y NO ESTAN INCLUIDOS EN EL REPOSITORIO. Esto se gestiona mediante el archivo .gitignore, donde exluimos los archivos .parquet de la carpeta data/raw/.

## Instrucciones para usar los datos
1. Descarga los archivos .parquet desde Kaggle: https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022/data
2. Coloca los archivos descargados en la carpeta data/raw
3. Asegúrate de que los nombres de los archivos coincidan con los utilizados en el código.

## Configuración

**Prerrequisitos**

Asegúrate de tener Python 3.11+ instalado en tu máquina. También necesitarás pip para instalar los paquetes de Python.

**Instalación**

Clona el repositorio del proyecto en tu máquina local.

Navega hasta el directorio del proyecto e instala los paquetes de Python requeridos:

```bash
pip install -r requirements.txt
```

**Crear una base de datos (si es necesario)**

Crea una nueva base de datos dentro del motor Postgres personalizando y ejecutando el siguiente comando: `$ createdb -h localhost -U <username> <db_name>`
Conéctate al motor Postgres para usar tu base de datos, manipular tablas y datos: `$ psql -h localhost -U <username> <db_name>`
NOTA: Recuerda revisar la información del archivo ./.env para obtener el nombre de usuario y db_name.

¡Una vez que estés dentro de PSQL podrás crear tablas, hacer consultas, insertar, actualizar o eliminar datos y mucho más!

**Variables de entorno**

Crea un archivo .env en el directorio raíz del proyecto para almacenar tus variables de entorno, como tu cadena de conexión a la base de datos:

```makefile
DATABASE_URL="your_database_connection_url_here"
```

## Ejecutando la Aplicación

Para ejecutar la aplicación, ejecuta el script app.py desde la raíz del directorio del proyecto:

```bash
python app.py
```

## Añadiendo Modelos

Para añadir clases de modelos SQLAlchemy, crea nuevos archivos de script de Python dentro del directorio models/. Estas clases deben ser definidas de acuerdo a tu esquema de base de datos.

Definición del modelo de ejemplo (`models/example_model.py`):

```py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id = Column(Integer, primary_key=True)
    name = Column(String)

```

## Trabajando con Datos

Puedes colocar tus conjuntos de datos brutos en el directorio data/raw, conjuntos de datos intermedios en data/interim, y los conjuntos de datos procesados listos para el análisis en data/processed.

Para procesar datos, puedes modificar el script app.py para incluir tus pasos de procesamiento de datos, utilizando pandas para la manipulación y análisis de datos.

## Contribuyentes

Esta plantilla fue construida como parte del [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) de 4Geeks Academy por [Alejandro Sanchez](https://twitter.com/alesanchezr) y muchos otros contribuyentes. Descubre más sobre [los programas BootCamp de 4Geeks Academy](https://4geeksacademy.com/us/programs) aquí.

Otras plantillas y recursos como este se pueden encontrar en la página de GitHub de la escuela.
