import mlflow
import pandas as pd
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# MLflow Tracking Server URI ayarla
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Modeli yükle
final_model = mlflow.pyfunc.load_model("models:/EarthquakePredictionModel/Production")

# Performansı izlemek için fonksiyon
def monitor_performance():
    while True:
        # Yeni veri ile tahmin yap
        data = {
            "Enlem": [39.9],
            "Boylam": [32.9],
            "Derinlik": [10],
            "Year": [2025],
            "Month": [5],
            "Day": [1]
        }

        # Veriyi DataFrame'e dönüştür
        input_data = pd.DataFrame(data)

        # Tahmin yap
        predictions = final_model.predict(input_data)

        # Performans metriği hesapla
        mae = mean_absolute_error([0.1], predictions)  # True value should be provided (example: 0.1)
        mse = mean_squared_error([0.1], predictions)
        r2 = r2_score([0.1], predictions)

        # Log metrics to MLflow
        mlflow.log_metric("monitor_mae", mae)
        mlflow.log_metric("monitor_mse", mse)
        mlflow.log_metric("monitor_r2", r2)

        print(f"Yeni Tahmin: {predictions}")
        print(f"Performance Metrics: MAE: {mae}, MSE: {mse}, R^2: {r2}")

        # Performansı belirli bir periyotta izlemek için 10 saniye bekle
        time.sleep(10)

if __name__ == "__main__":
    monitor_performance()
