import streamlit as st
import joblib
import pandas as pd


# Cargar el modelo
model = joblib.load("my_final_model.pkl")

# Título de la aplicación
st.title("Predicción de Precios de Viviendas")

# Ingresar datos de entrada
st.header("Ingrese las características de la vivienda")

# Ejemplo de campos de entrada (ajusta según las características de tu modelo)
housing_median_age = st.number_input("Edad media de las viviendas", min_value=1, max_value=10)
pop_per_hhold = st.number_input("Población por hogar", min_value=1, max_value=10)
bedrooms_per_room = st.number_input("Número de dormitorios por habitación", min_value=1, max_value=10)

# Otros campos de entrada...
# (Agrega más campos según las características de tu modelo)

# Botón para realizar la predicción
if st.button("Predecir Precio"):
    # Crear un DataFrame con las entradas del usuario
    input_data = pd.DataFrame({
        "housing_median_age": [housing_median_age],
        "pop_per_hhold": [pop_per_hhold],
        "bedrooms_per_room": [bedrooms_per_room],
        # Agrega aquí otros campos según tu modelo
    })

    # Realizar la predicción
    prediction = model.predict(input_data)

    # Mostrar el resultado
    st.success(f"El precio estimado de la vivienda es: ${prediction[0]:,.2f}")
