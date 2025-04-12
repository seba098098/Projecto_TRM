from prefect import flow, task
import pandas as pd
import joblib
import yfinance as yf
import requests
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt
import warnings
import subprocess

warnings.filterwarnings('ignore')
# 1. Carga de datos
# Descargar las noticias / Crypto panic NewsAPI
@task
def download_bitcoin_news():
    # Tu API Key de NewsAPI
    '''API_KEY = 'a8a6fc010f8445b7b236369fb7bf273d'

    # Fechas
    hoy = datetime.now()
    hace_30_dias = hoy - timedelta(days=30)

    # Convertir fechas al formato requerido por la API
    from_date = hace_30_dias.strftime('%Y-%m-%d')
    to_date = hoy.strftime('%Y-%m-%d')

    # Endpoint de NewsAPI
    url = 'https://newsapi.org/v2/everything'

    # Parámetros de búsqueda
    params = {
        'q': 'bitcoin',
        'from': from_date,
        'to': to_date,
        'language': 'es',
        'sortBy': 'publishedAt',
        'pageSize': 2,
        'apiKey': API_KEY
    }

    # Hacer la solicitud
    response = requests.get(url, params=params)
    data = response.json()

    # Extraer y mostrar resultados
    if data['status'] == 'ok':
        articles = data['articles']
        df = pd.DataFrame(articles)
        df = df[['publishedAt', 'title', 'description', 'url']]
        df.to_csv('noticias_bitcoin.csv', index=False, encoding='utf-8-sig')
        print(f"Se guardaron {len(df)} noticias en 'noticias_bitcoin.csv'")
    else:
        print("Error:", data)
    df.to_csv('Datos/noticias_bitcoin.csv', index=False, encoding='utf-8-sig')'''
    # Cargar el dataset original
    df = pd.read_csv("Datos/noticias_bitcoin_top3_diarias.csv")
    df = df.rename(columns={
    "fecha": "publishedAt",
    "titulo": "title",
    "descripcion": "description"
})

    return df
# Descargar los datos de Bitcoin / yfiannce
@task
def download_bitcoin_finance_data():
    """
    Descarga datos históricos de Bitcoin usando yfinance.
    Aquí usamos un período de 1 día y un intervalo diario,
    pero puedes ajustar estos parámetros según tus necesidades.
    """
    # Obtener laS noticias de hace 30 días
    
    btc_data = yf.download("BTC-USD", period="30d", interval="1d")
    btc_data.columns = btc_data.columns.get_level_values(0)
    df_simple = btc_data[['Close']].reset_index() 
    df_simple.columns = ['Fecha', 'Close']



    return df_simple
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
    df['score'] = df['title'].apply(get_sentiment_score)
    df_final = df.rename(columns={"publishedAt": "fecha", "title": "titulo"})
    df_final = df_final[['fecha', 'score']]
    df_final=df_final.groupby('fecha').agg({'score': 'mean'}).reset_index()
    # Agrupar y calcular la media por fecha


# Mostrar resultados

    return df_final
# Unir datasets noticias y precio 
@task
def unir_datasets():
    df_noticias = procesar_noticias_hoy()
    df_precios = download_bitcoin_finance_data()
    # Convertir la columna 'fecha' a tipo datetime
    df_noticias['fecha'] = pd.to_datetime(df_noticias['fecha'])
    df_precios['Fecha'] = pd.to_datetime(df_precios['Fecha'])

    # Unir los datasets por fecha
    df_unido = pd.merge(df_precios, df_noticias, left_on='Fecha', right_on='fecha', how='left')
    df_unido['score'] = df_unido['score'].fillna(0)
    df_unido = df_unido[['Fecha', 'score', 'Close']]


    return df_unido
# 3. Entrenar el modelo de bitcoin y trackearlo con ml flow
def entrenar_modelo():
    # Cargar el modelo pre-entrenado
    df=unir_datasets()
    # Separar características y etiquetas
    X = df[['score']]
    y = df['Close']
    # Entrenar el modelo prophet
    model = Prophet()
    model.add_regressor('score')
    model.fit(df.rename(columns={'Fecha': 'ds', 'Close': 'y'}))
    # Guardar el modelo entrenado
    return model,X,y
# Guardar el modelo entrenado
@task

@task
def git_auto_commit(file_path):
    try:
        # Configura usuario de Git (solo necesario la primera vez)
        subprocess.run(["git", "config", "--global", "user.name", "santiloc-hub"])
        subprocess.run(["git", "config", "--global", "user.email", "santiloc23456@gmail.com"])
        
        # Haz commit y push
        subprocess.run(["git", "add", file_path])
        subprocess.run(["git", "commit", "-m", f"Resultados automáticos {datetime.now().strftime('%Y-%m-%d %H:%M')}"])
        subprocess.run(["git", "push"])
        return True
    except Exception as e:
        print(f"Error en git_auto_commit: {e}")
        return False
# 4. Hacer la predicción
@task
def predecir_precio():
    # Cargar el modelo entrenado
    model,X,Y = entrenar_modelo()
    # Crear un DataFrame para la predicción
    future = model.make_future_dataframe(periods=7)
    future['score'] = 0
    # Hacer la predicción
    forecast = model.predict(future)
    # Obtener el precio predicho
    # Seleccionar los últimos 7 días de predicción con sus intervalos
    predicciones_7dias = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

    # Establecer la fecha como índice
    predicciones_7dias = predicciones_7dias.set_index('ds')

    # Renombrar columnas (opcional)
    predicciones_7dias = predicciones_7dias.rename(columns={
        'yhat': 'prediccion',
        'yhat_lower': 'min_confianza',
        'yhat_upper': 'max_confianza'
    })
   



    return predicciones_7dias
    


# 6. Crear un flujo de trabajo que ejecute todas las tareas en orden a las 9:00 am
@flow
def flujo_prediccion_bitcoin():
    #Guardar resultados
    df = predecir_precio()
    fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ruta_csv = f"Work_Flow/Predicciones/predicciones_bitcoin_{fecha}.csv"
    df.to_csv(ruta_csv, index=False)
     # Commit automático
    git_auto_commit(ruta_csv)
if __name__ == "__main__":
    flujo_prediccion_bitcoin()
