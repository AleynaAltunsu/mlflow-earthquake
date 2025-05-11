import mlflow
from mlflow.tracking import MlflowClient

# MLflow sunucu adresi (eğer özel bir URI kullanıyorsan bunu belirt)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Model adı (train.py'de kayıt edilenle aynı olmalı)
model_name = "EarthquakePredictionModel"

# Client oluştur
client = MlflowClient()

# Son kayıtlı model versiyonunu çek (Henüz "stage" atanmamış olanlardan)
latest_versions = client.get_latest_versions(name=model_name)


# Eğer henüz kayıtlı model yoksa uyar
if not latest_versions:
    print(f"Henüz Model Registry'e kayıtlı bir model versiyonu yok.")
else:
    latest_version = latest_versions[0]
    print(f"Son model versiyonu: {latest_version.version}")

    # Production aşamasına geçir
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production"
    )
    print(f"Model {latest_version.version} başarıyla 'Production' aşamasına geçirildi.")
