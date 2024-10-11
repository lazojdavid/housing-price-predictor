import streamlit as st
import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load("my_final_model.pkl")

# Título de la aplicación
st.title("Predicción de Precios de Viviendas con respecto al Área")

# Ingresar datos de entrada
st.header("Ingrese las características de la vivienda")

# Campos de entrada
median_income = st.number_input("Ingreso medio del área", min_value=0.0)
inland = st.selectbox("¿Está cerca a la costa?", options=["Sí", "No"])
pop_per_hhold = st.number_input("Población por hogar", min_value=0.0)
bedrooms_per_room = st.number_input("Número de dormitorios por habitación", min_value=0.0)
longitude = st.number_input("Longitud", min_value=-180.0, max_value=180.0, format="%.6f")
latitude = st.number_input("Latitud", min_value=-90.0, max_value=90.0, format="%.6f")
rooms_per_hhold = st.number_input("Habitaciones por hogar", min_value=0.0)
housing_median_age = st.number_input("Edad media de las viviendas", min_value=0)
total_rooms = st.number_input("Número total de habitaciones", min_value=0)
total_bedrooms = st.number_input("Número total de dormitorios", min_value=0)
population = st.number_input("Población total", min_value=0)
households = st.number_input("Número total de hogares", min_value=0)

# Proximidad al océano
ocean_proximity = st.selectbox("Proximidad al océano", options=["<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND", "INLAND"])

# Botón para realizar la predicción
if st.button("Predecir Precio"):
    # Crear un DataFrame con todas las entradas del usuario
    input_data = pd.DataFrame({
        "median_income": [median_income],
        "INLAND": [1 if inland == "Sí" else 0],
        "pop_per_hhold": [pop_per_hhold],
        "bedrooms_per_room": [bedrooms_per_room],
        "longitude": [longitude],
        "latitude": [latitude],
        "rooms_per_hhold": [rooms_per_hhold],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "ocean_proximity_<1H OCEAN": [1 if ocean_proximity == "<1H OCEAN" else 0],
        "ocean_proximity_NEAR_OCEAN": [1 if ocean_proximity == "NEAR OCEAN" else 0],
        "ocean_proximity_NEAR_BAY": [1 if ocean_proximity == "NEAR BAY" else 0],
        "ocean_proximity_ISLAND": [1 if ocean_proximity == "ISLAND" else 0],
    })

    # Realizar la predicción
    prediction = model.predict(input_data)

    # Mostrar el resultado
    st.success(f"El precio estimado de la vivienda es: ${prediction[0]:,.2f}")
