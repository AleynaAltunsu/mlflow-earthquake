import requests
import json

# Flask API'ye POST isteği gönder
url = "http://127.0.0.1:5000/predict"
data = {
    "Enlem": [39.9],
    "Boylam": [32.9],
    "Derinlik": [10],
    "Year": [2025],
    "Month": [5],
    "Day": [1]
}

# API'ye isteği gönder
response = requests.post(url, json=data)

# Yanıtı kontrol et
if response.status_code == 200:
    try:
        prediction = response.json()
        print("Tahmin:", prediction)
    except requests.exceptions.JSONDecodeError as e:
        print("JSON Decode Hatası:", e)
else:
    print(f"API Hatası: {response.status_code}")
