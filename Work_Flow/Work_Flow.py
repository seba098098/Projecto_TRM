from prefect import flow, task
import pandas as pd
import joblib
import yfinance as yf
import requests
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 1. Carga de datos
# Descargar las noticias / Crypto panic
@task
def download_bitcoin_news():
    """
    Descarga noticias de Bitcoin desde la API de CryptoPanic y filtra las de los últimos 5 días.
    """
    API_KEY = "4760b36976ccd9d3da9909e92509ca3f67f3f19b"  # 🔁 Reemplaza esto por tu clave real
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": API_KEY,
        "currencies": "BTC",
        "public": "true"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error al obtener datos: {response.status_code}")

    data = response.json()
    results = data.get("results", [])
    if not results:
        print("No se encontraron noticias.")
        return pd.DataFrame()

    # Convertir a DataFrame
    df = pd.DataFrame(results)
    df_final=df.rename(columns={"title": "titulo", "published_at": "fecha"})
    df_final=df_final[["titulo", "fecha"]]
    #df_final.to_csv("../Datos/noticias_bitcoin_hoy.csv", index=False)

    return df_final 
# Descargar los datos de Bitcoin / yfiannce
@task
def download_bitcoin_finance_data():
    """
    Descarga datos históricos de Bitcoin usando yfinance.
    Aquí usamos un período de 1 día y un intervalo diario,
    pero puedes ajustar estos parámetros según tus necesidades.
    """
    # Obtener laS noticias de hace 1 días
    
    btc_data = yf.download("BTC-USD", period="1d", interval="1d")
    
    # Convertir el índice a una columna y renombrar las columnas
    
    btc_data.reset_index(inplace=True)
    btc_data.columns = [col[0].lower() for col in btc_data.columns]
    btc_data.columns = [col.capitalize() for col in btc_data.columns]
    
    # Cargar dataset original
    
    df_btc=pd.read_csv("Datos/btc_USD_historic.csv")
    
    # Concatenar los datasets y eliminar duplicados
    
    df_btc["Date"]=pd.to_datetime(df_btc["Date"])
    btc_data=pd.concat([df_btc,btc_data])
    #btc_data=btc_data[1:]
    btc_data = btc_data.drop_duplicates(subset="Date", keep="last")
    
    # Guardar el dataset actualizado

    btc_data.to_csv("Datos/btc_USD_historic.csv", index=False)

    return btc_data

#Procesar noticias de hoy
@task
def procesar_noticias_hoy():
    df=download_bitcoin_news()
    # Cargar modelo y tokenizer
    model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Función para calcular el sentimiento de una frase
    def get_sentiment_score(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).detach().numpy()[0]
        neg, neu, pos = probs
        score = -1 * neg + 0 * neu + 1 * pos
        return score

# Aplicar a la columna de títulos
    df['score'] = df['titulo'].apply(get_sentiment_score)
    

# Mostrar resultados

    return df
# Agrupar noticias por fecha y calcular el promedio de sentimiento
@task
def agrupar_noticias_por_fecha():
    df = procesar_noticias_hoy()
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Agrupar por fecha y calcular el promedio del sentimiento
    df_grouped = df.groupby(df['fecha'].dt.date).agg({'score': 'mean'}).reset_index()
    return df_grouped
# Eliminar valor de close con fecha mas antigua y concatenar al dataset original
@task
# Guradar el dataset actualizado
def guardar_dataset_actualizado():
    df_grouped = agrupar_noticias_por_fecha()
    df_news=pd.read_csv("Datos/noticias_bitcoin_sentimientos.csv")
    df_news=df_news[['fecha','score']].drop_duplicates(subset='fecha', keep='last')
    return df_news.columns,df_grouped.columns
    
# 2. Entrenar el modelo de noticas

# 3. Entrenar el modelo de bitcoin y trackearlo con ml flow

# 4. Hacer la prediccion

# 5. Guardar la prediccion en un archivo CSV

# 6. Crear un flujo de trabajo que ejecute todas las tareas en orden a las 9:00 am
@flow
def flujo_prediccion_bitcoin():
    print(guardar_dataset_actualizado())
    
if __name__ == "__main__":
    flujo_prediccion_bitcoin()
