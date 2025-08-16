Perfecto 🚀, aquí tienes el **README completo** en el mismo formato que me diste, pero ya con **GraphQL** y **gRPC** añadidos en la sección de componentes y accesos:

---

# Predicción de Preferencia de Canciones en Spotify

### Miembros del Equipo del Proyecto

* **Jonathan Cagua**
* **Juan Pablo Alianak**

## Objetivo

El objetivo de este análisis es predecir si una nueva canción resultará atractiva para la persona que utiliza activamente esta playlist. Para lograrlo, se consideran diversas características musicales de las canciones como variables de entrada para el modelo:

1. **Speechiness (0.0 - 1.0)**
   Indica la presencia de elementos hablados en la pista. Un valor cercano a 1.0 sugiere que la pista es predominantemente hablada, mientras que valores bajos corresponden a pistas más musicales.

2. **Energy (0.0 - 1.0)**
   Representa la intensidad y actividad de la pista. Un valor alto indica una canción enérgica, mientras que un valor bajo corresponde a una más relajada.

3. **Danceability (0.0 - 1.0)**
   Mide qué tan apta es la canción para bailar, considerando factores como ritmo, estabilidad y regularidad. Un valor más alto indica mayor facilidad para bailar.

4. **Acousticness (0.0 - 1.0)**
   Refleja la probabilidad de que la pista sea acústica. Un valor cercano a 1.0 indica un alto grado de elementos acústicos, mientras que valores bajos sugieren mayor uso de instrumentos electrónicos o producción digital.

Estas variables capturan distintas dimensiones de las canciones, proporcionando una base sólida para evaluar y predecir su potencial éxito dentro de una playlist específica.

![alt text](image-4.png)

## Fuente del Dataset

El dataset utilizado en este análisis proviene de la colección de canciones de Spotify. Puedes acceder al dataset desde [este enlace](https://drive.google.com/file/d/13DDhnS2FoXN-xqWM9PWryQyUBgRTqcKg/view?usp=sharing).

---

## Componentes del Proyecto

Este proyecto involucra los siguientes servicios y herramientas clave:

1. **Airflow**: Orquesta el pipeline ETL para procesar los datos de Spotify.
2. **MLflow**: Realiza el seguimiento de los experimentos de machine learning y registra datasets.
3. **MinIO**: Proporciona almacenamiento de objetos compatible con S3 para datos y artefactos de MLflow.
4. **FastAPI / GraphQL / gRPC**:

   * **FastAPI** expone endpoints REST tradicionales para la inferencia y gestión de datasets.
   * **GraphQL** ofrece un endpoint flexible para consultar y consumir predicciones.
   * **gRPC** provee un canal de comunicación eficiente y de alto rendimiento para integrar el modelo con otros sistemas.
5. **Streamlit**: Ofrece una interfaz de visualización e interacción para los usuarios.

![Arquitectura](Diagram.jpg "Componentes del Proyecto")

---

## Instrucciones de Configuración Local

### Requisitos Previos

Asegúrate de tener instalados:

* Docker y Docker Compose
* Python 3.8+
* AWS CLI (para configuración de MinIO si es necesario)

### Variables de Entorno

Usa la siguiente configuración en un archivo `.env` para Docker Compose:

```env
# Configuración Airflow
AIRFLOW_UID=50000
AIRFLOW_GID=0
AIRFLOW_PROJ_DIR=./airflow
AIRFLOW_PORT=8083
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# Configuración PostgreSQL
PG_USER=airflow
PG_PASSWORD=airflow
PG_DATABASE=airflow
PG_PORT=5444

# Configuración MLflow
MLFLOW_PORT=5006
MLFLOW_S3_ENDPOINT_URL=http://s3:9000

# Configuración MinIO
MINIO_ACCESS_KEY=minio
MINIO_SECRET_ACCESS_KEY=minio123
MINIO_PORT=9008
MINIO_PORT_UI=9009
MLFLOW_BUCKET_NAME=mlflow
DATA_REPO_BUCKET_NAME=data

# Configuración FastAPI
FASTAPI_PORT=8803

# Configuración Streamlit
STREAMLIT_PORT=8504
```

---

## Detalles de Acceso a los Servicios

### 1. Airflow

* **Descripción**: Administra y monitorea el pipeline ETL.
* **URL**: [http://localhost:8083](http://localhost:8083)
* **Credenciales**:

  * Usuario: `airflow`
  * Contraseña: `airflow`

### 2. MLflow

* **Descripción**: Realiza seguimiento de experimentos y registra datasets.
* **URL**: [http://localhost:5006](http://localhost:5006)

### 3. MinIO

* **Descripción**: Proporciona almacenamiento de objetos para datasets y artefactos.
* **Consola URL**: [http://localhost:9009](http://localhost:9009)
* **Credenciales**:

  * Access Key: `minio`
  * Secret Key: `minio123`

### 4. FastAPI

* **Descripción**: Expone endpoints API REST para predicciones y gestión de datasets.
* **URL**: [http://localhost:8803/docs#/](http://localhost:8803/docs#/)

### 5. GraphQL

* **Descripción**: Endpoint flexible para consultas y predicciones personalizadas.
* **URL**: [http://localhost:8803/graphql](http://localhost:8803/graphql)

### 6. gRPC

* **Descripción**: Servicio de alto rendimiento para inferencias, ideal para integraciones entre sistemas.
* **Puerto**: `50051`

### 7. Streamlit

* **Descripción**: Dashboard interactivo para explorar datos y resultados.
* **URL**: [http://localhost:8504](http://localhost:8504)

---

## Resumen del Flujo de Trabajo

### Pasos del Pipeline ETL:

1. **Ingesta de Datos**:

   * Descarga el dataset desde Google Drive.
   * Lo almacena en un bucket S3 usando MinIO.

2. **Ingeniería de Características**:

   * Escala variables numéricas (`duration`, `tempo`, `loudness`) usando `MinMaxScaler`.
   * Retiene características clave para el modelado (`speechiness`, `energy`, `danceability`, `acousticness`).
   * Almacena el dataset procesado nuevamente en MinIO.

3. **División del Dataset**:

   * Divide el dataset en conjuntos de entrenamiento y prueba (70/30) usando muestreo estratificado.
   * Guarda los conjuntos en S3.

4. **Registro del Dataset**:

   * Registra metadatos y estadísticas del dataset (media, desviación estándar) en S3 y MLflow.

---

## Ejecución del Proyecto

### Paso 1: Iniciar los Servicios

Ejecuta el siguiente comando para iniciar todos los servicios con Docker Compose:

```bash
docker compose --profile all up
```

### Paso 2: Acceder a los Servicios

* Acceder a **Airflow**

![alt text](image.png)

* Acceder a **MLflow**

![alt text](image-1.png)

* Acceder a **MinIO**

![alt text](image-2.png)

* Acceder a **FastAPI**

![alt text](image-3.png)

* Acceder a **Streamlit**

![alt text](image-4.png)

### Paso 3: Ejecutar el DAG de ETL

* Inicia sesión en la UI de Airflow.
* Lanza el DAG llamado `process_etl_spotify_data`.

### Paso 4: Monitorear y Explorar

* Verifica logs y estado en la UI de Airflow.
* Revisa datasets y tracking de experimentos en MLflow.
* Usa los endpoints de FastAPI, GraphQL o gRPC para predicciones.
* Explora los datos y resultados en Streamlit.
