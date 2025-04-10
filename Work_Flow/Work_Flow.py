from prefect import flow, task
import pandas as pd
import joblib
import yfinance as yf
import requests
from datetime import datetime, timedelta
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
@task
def download_bitcoin_finance_data():
    """
    Descarga datos históricos de Bitcoin usando yfinance.
    Aquí usamos un período de 5 días y un intervalo diario,
    pero puedes ajustar estos parámetros según tus necesidades.
    """
    # Obtener laS noticias de hace 5 días
    
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

# Eliminar noticias con la fecha mas antigua y concatenar al dataset original
@task
def eliminar_noticias_duplicadas():
    df=download_bitcoin_news()

    return df

# Eliminar valor de close con fecha mas antigua y concatenar al dataset original
@task

# 2. Entrenar el modelo de noticas

# 3. Entrenar el modelo de bitcoin y trackearlo con ml flow

# 4. Hacer la prediccion

# 5. Guardar la prediccion en un archivo CSV

# 6. Crear un flujo de trabajo que ejecute todas las tareas en orden a las 9:00 am
@flow
def flujo_prediccion_bitcoin():
    print(download_bitcoin_finance_data())
    ##download_bitcoin_news().info()
if __name__ == "__main__":
    flujo_prediccion_bitcoin()
