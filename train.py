             # train.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import warnings
warnings.filterwarnings("ignore")

# Veri Yükleme
df = pd.read_csv('C:/Users/aleyn/OneDrive/Masaüstü/earthquake-mlflow/data/turkey_earthquakes.csv', delimiter=';')

# Ön İşleme
df['Datetime'] = pd.to_datetime(df['Olus tarihi'] + ' ' + df['Olus zamani'], errors='coerce')
df = df.dropna(subset=['Datetime', 'Enlem', 'Boylam', 'Derinlik', 'ML'])
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day

# Özellikler ve Hedef
features = ['Enlem', 'Boylam', 'Derinlik', 'Year', 'Month', 'Day']
target = 'ML'
X = df[features]
y = df[target]

# Eğitim / Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparametre Optimizasyonu
def objective(params):
    with mlflow.start_run(nested=True):
        model = RandomForestRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        return {'loss': mse, 'status': STATUS_OK}

# Arama Alanı
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)
}

# MLflow Ayarı
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("EarthquakeMagnitude_Hyperopt")

# Ana Eğitim Süreci
with mlflow.start_run(run_name="Hyperopt_RandomForest") as run:
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=25, trials=trials)
    mlflow.log_params(best)

    final_model = RandomForestRegressor(
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        min_samples_split=int(best['min_samples_split']),
        random_state=42
    )
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("final_mae", mae)
    mlflow.log_metric("final_mse", mse)
    mlflow.log_metric("final_r2", r2)

    # Modeli kaydet
    mlflow.sklearn.log_model(final_model, "model")

    # Modeli registry'ye kaydet
    logged_model_uri = f"runs:/{run.info.run_id}/model"
    result = mlflow.register_model(model_uri=logged_model_uri, name="EarthquakePredictionModel")

    # Production'a geçir
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="EarthquakePredictionModel",
        version=result.version,
        stage="Production"
    )

#################### Flask API ####################

from flask import Flask, request, jsonify
import mlflow.pyfunc

# Flask başlat
app = Flask(__name__)

# Production modelini yükle
model = mlflow.pyfunc.load_model("models:/EarthquakePredictionModel/Production")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame(data)
    predictions = model.predict(input_data)
    return jsonify(predictions.tolist())

#################### Performans İzleme ####################
import time

def monitor_performance(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_metric("monitor_mae", mae)
    mlflow.log_metric("monitor_mse", mse)
    mlflow.log_metric("monitor_r2", r2)

    return {"mae": mae, "mse": mse, "r2": r2}

