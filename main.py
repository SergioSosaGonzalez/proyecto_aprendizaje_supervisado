from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Cargar el modelo
modelo_protocolo = joblib.load('modelo_supervisado_protocolo.pkl')
modelo_kmeans = joblib.load('modelo_kmeans_deposito.pkl')
escalador_kmeans = joblib.load('escalador_kmeans.pkl')

app = FastAPI(title="Clasificador de Productos para protocolo y deposito")

class Product(BaseModel):
    embalaje: int
    ancho_cm: float
    largo_cm: float
    alto_cm: float
    peso_kg: float
    procedencia: str
    manipulacion: str
    temperatura: str

@app.get("/")
def read_root():
    return {"Hello": "Welcome to the product classifier"}


@app.post("/clasificar")
def clasificar_producto(p: Product):
    df_input = pd.DataFrame([p.dict()])
    protocolo = modelo_protocolo.predict(df_input)[0]
    df_input['volumen_cm3'] = df_input['ancho_cm'] * df_input['largo_cm'] * df_input['alto_cm']
    cluster_input = df_input[['volumen_cm3', 'peso_kg', 'manipulacion', 'temperatura']]
    cluster_input = pd.get_dummies(cluster_input, columns=['manipulacion', 'temperatura'], drop_first=True)
    for col in ['manipulacion_normal', 'temperatura_refrigerado']:
        if col not in cluster_input.columns:
            cluster_input[col] = 0
    cluster_input = cluster_input[['volumen_cm3', 'peso_kg', 'manipulacion_normal', 'temperatura_refrigerado']]
    x_scaled = escalador_kmeans.transform(cluster_input)
    cluster = modelo_kmeans.predict(x_scaled)[0]
    deposito = f"Deposito_{cluster+1}"
    return {"deposito": deposito, "protocolo": protocolo}