# Iris Species Classifier - MLOps Experiment 8

End-to-end MLOps project on a small ML model: a Random Forest that predicts the
species of an Iris flower (setosa, versicolor or virginica) from four petal and
sepal measurements, served as a Streamlit web app.

**Arnav Mhatre - E002 - BTI DS Sem 11 (E1 batch)**

## Pipeline

| Stage | Tool | Files |
|---|---|---|
| Model | scikit-learn Random Forest (test accuracy 96.7%) | `train.py`, `data/iris.csv`, `model.pkl` |
| Version control | Git + GitHub | this repo |
| CI | GitHub Actions (lint, train, test, Docker build, smoke test) | `.github/workflows/ci.yml`, `test_model.py` |
| Container | Docker | `Dockerfile` |
| Orchestration | Kubernetes (Docker Desktop) | `k8s/deployment.yaml`, `k8s/service.yaml` |
| Monitoring | Prometheus + Grafana | `docker-compose.yml`, `monitoring/` |
| Deployment | Streamlit Community Cloud | `app.py`, `requirements.txt` |

## Run locally

```bash
pip install -r requirements.txt
python train.py
pytest -v
streamlit run app.py
```

## Docker

```bash
docker build -t iris-app:E002 .
docker run -d -p 8501:8501 -p 8001:8001 --name iris-app iris-app:E002
```

## Kubernetes

```bash
kubectl apply -f k8s/
kubectl get pods
kubectl port-forward service/iris-service 8502:8501
```

## Monitoring

```bash
docker compose up -d --build
```

- App: http://localhost:8501
- Metrics: http://localhost:8001/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (dashboard "Iris Classifier - Model Monitoring" is loaded automatically)

Metrics exposed by the app: `iris_predictions_total` (by species and source),
`iris_prediction_latency_seconds`, `iris_prediction_confidence`,
`iris_input_feature_value` (last input per feature) and `iris_model_accuracy`.
The sidebar button "Send simulated requests" sends a batch of predictions so the
dashboard has data to show.

## Dataset

Iris flower dataset (R. A. Fisher, 1936), as shipped with scikit-learn.
150 flowers, 50 per species, 4 numeric features in cm, no missing values.
