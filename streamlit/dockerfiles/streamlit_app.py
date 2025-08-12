import streamlit as st
import requests

# URL del servicio FastAPI
API_URL = "http://fastapi:8800/predict/"

# Título de la app
st.title("Análisis de canciones: Predicción de éxito")

st.markdown("""
Esta aplicación permite analizar las características de una canción y predecir si será exitosa o no. 
Por favor, ingresa los valores de las siguientes características musicales.
""")

# Entrada de datos para las características de la canción
speechiness = st.number_input("Speechiness (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.0001, format="%.4f")
energy = st.number_input("Energy (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.0001, format="%.4f")
danceability = st.number_input("Danceability (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.0001, format="%.4f")
acousticness = st.number_input("Acousticness (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.0001, format="%.4f")

# Botón para realizar la predicción
if st.button("Analizar canción"):
    # Crear el cuerpo de la solicitud en formato JSON
    song_features = {
        "features": {  
            "speechiness": speechiness,
            "energy": energy,
            "danceability": danceability,
            "acousticness": acousticness
        }
    }

    try:
        # Enviar la solicitud POST al servicio FastAPI
        response = requests.post(API_URL, json=song_features)
        
        # Verificar el código de estado de la respuesta
        if response.status_code == 200:
            # Mostrar los resultados de la predicción
            prediction_data = response.json()
            prediction = prediction_data.get("str_output", "Sin información")
            
            if prediction.lower() == "successful":
                st.success("¡La canción tiene características prometedoras para ser exitosa!")
            else:
                st.warning("La canción puede no ser exitosa según las características proporcionadas.")
        else:
            st.error(f"Error: No se pudo analizar la canción. Código de estado: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión con el servicio FastAPI: {e}")

