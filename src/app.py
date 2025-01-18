# Predicción de Retrasos en Vuelos - Notebook Mejorado
# ======================================================

# 0. Imports
# ----------

# Basic imports
import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

# Functional imports
import json
import pyarrow
from pickle import dump
import gc
from datetime import datetime

# Preprocessing imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Model imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# Configuraciones
pd.set_option('display.max_columns', None)
np.random.seed(42)

# 1. Carga de Datos
# ----------------

def load_data():
    """Carga y realiza preprocesamiento inicial de los datos"""
    print("Cargando datos...")
    df = pd.read_parquet("../data/raw/Combined_Flights_2022.parquet", engine="pyarrow")
    
    # Seleccionar columnas relevantes
    columns_to_keep = [
        'DayOfWeek', 'Month', 'Quarter', 'DayofMonth',
        'DepDelayMinutes', 'DepTime', 'CRSDepTime',
        'Distance', 'Airline', 'OriginStateName',
        'DestStateName', 'OriginCityName', 'DestCityName'
    ]
    
    df = df[columns_to_keep]
    print(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
    return df

# 2. Análisis Exploratorio de Datos (EDA)
# -------------------------------------

def perform_eda(df):
    """Realiza análisis exploratorio de datos"""
    print("\nAnálisis Exploratorio de Datos:")
    
    # Estadísticas básicas
    print("\nEstadísticas descriptivas de variables numéricas:")
    print(df.describe())
    
    # Valores nulos
    print("\nValores nulos por columna:")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0])
    
    # Análisis de la variable objetivo
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='DepDelayMinutes', bins=50)
    plt.title('Distribución de Retrasos')
    plt.show()
    
    # Correlaciones
    plt.figure(figsize=(12, 8))
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlaciones')
    plt.show()

# 3. Feature Engineering
# --------------------

def create_cyclical_features(df):
    """Crea características cíclicas para variables temporales"""
    print("\nCreando características cíclicas...")
    
    # Convertir DepTime a hora del día (0-24)
    df['HourBlock'] = df['DepTime'].apply(lambda x: int(str(int(x)).zfill(4)[:2]) + 
                                        int(str(int(x)).zfill(4)[2:])/60)
    
    # Features cíclicos
    for col, max_val in zip(['HourBlock', 'DayOfWeek', 'Month'], 
                          [24, 7, 12]):
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    
    return df

def create_interaction_features(df):
    """Crea características de interacción"""
    print("Creando características de interacción...")
    
    df['Distance_Hour'] = df['Distance'] * df['HourBlock']
    df['Distance_DayOfWeek'] = df['Distance'] * df['DayOfWeek']
    df['Hour_DayOfWeek'] = df['HourBlock'] * df['DayOfWeek']
    
    return df

def handle_outliers(df, columns, n_std=3):
    """Maneja outliers usando el método de z-score"""
    print("\nManejando outliers...")
    df_clean = df.copy()
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df_clean[col] = df_clean[col].clip(lower=mean - n_std * std, 
                                         upper=mean + n_std * std)
    
    return df_clean

# 4. Preprocesamiento
# ------------------

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """Crea pipeline de preprocesamiento"""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# 5. Modelado
# ----------

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Entrena y evalúa múltiples modelos"""
    models = {
        'XGBoost': XGBRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEntrenando {name}...")
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Métricas
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X_train, y_train, 
                                  cv=5, scoring='neg_mean_squared_error')
        results[name]['CV_RMSE'] = np.sqrt(-cv_scores.mean())
    
    return results, models

def optimize_best_model(best_model_name, X_train, y_train):
    """Optimiza hiperparámetros del mejor modelo"""
    if best_model_name == 'XGBoost':
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3]
        }
        model = XGBRegressor(random_state=42)
    elif best_model_name == 'RandomForest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        model = RandomForestRegressor(random_state=42)
    else:  # LightGBM
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7]
        }
        model = LGBMRegressor(random_state=42)
    
    grid_search = GridSearchCV(model, param_grid, cv=5, 
                             scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

# 6. Función Principal
# ------------------

def main():
    # 1. Cargar datos
    df = load_data()
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Feature Engineering
    df = create_cyclical_features(df)
    df = create_interaction_features(df)
    df = handle_outliers(df, ['Distance', 'DepDelayMinutes'])
    
    # 4. Preparación de datos
    X = df.drop('DepDelayMinutes', axis=1)
    y = df['DepDelayMinutes']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Definir características
    numerical_features = ['Distance', 'HourBlock', 'Distance_Hour', 
                         'Distance_DayOfWeek', 'Hour_DayOfWeek', 
                         'HourBlock_sin', 'HourBlock_cos',
                         'DayOfWeek_sin', 'DayOfWeek_cos',
                         'Month_sin', 'Month_cos']
    
    categorical_features = ['Airline', 'OriginStateName', 'DestStateName']
    
    # Crear y aplicar pipeline
    preprocessor = create_preprocessing_pipeline(
        numerical_features, categorical_features
    )
    
    # Preprocesar datos
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 5. Modelado
    results, models = train_and_evaluate_models(
        X_train_processed, X_test_processed, y_train, y_test
    )
    
    # Encontrar mejor modelo
    best_model_name = min(results.items(), 
                         key=lambda x: x[1]['RMSE'])[0]
    
    # Optimizar mejor modelo
    best_model, best_params = optimize_best_model(
        best_model_name, X_train_processed, y_train
    )
    
    # Guardar modelo y resultados
    print("\nGuardando modelo y resultados...")
    dump(best_model, open('../models/best_model.pkl', 'wb'))
    dump(preprocessor, open('../models/preprocessor.pkl', 'wb'))
    
    with open('../models/model_metrics.json', 'w') as f:
        json.dump(results, f)
    
    return best_model, preprocessor, results

if __name__ == "__main__":
    best_model, preprocessor, results = main()
    print("\nResultados finales:")
    print(json.dumps(results, indent=2))