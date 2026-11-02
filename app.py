# app.py - Streamlit web app for the Iris species classifier (MLOps Experiment 8)
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from prometheus_client import Counter, Gauge, Histogram, start_http_server

st.set_page_config(page_title="Iris Species Classifier", page_icon="🌸", layout="wide")


# Load the trained model and the dataset only once
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/iris.csv")


# Start the Prometheus metrics server only once (http://localhost:8001/metrics)
@st.cache_resource
def start_metrics(accuracy, f1):
    try:
        start_http_server(8001)
    except OSError:
        pass  # port already in use, the app still works without metrics
    metrics = {
        "predictions": Counter("iris_predictions_total", "Number of predictions made",
                               ["species", "source"]),
        "latency": Histogram("iris_prediction_latency_seconds", "Time taken for one prediction",
                             buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]),
        "confidence": Histogram("iris_prediction_confidence", "Probability of the predicted species",
                                buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]),
        "feature": Gauge("iris_input_feature_value", "Last input value for each feature (cm)",
                         ["feature"]),
        "accuracy": Gauge("iris_model_accuracy", "Test accuracy of the deployed model"),
        "f1": Gauge("iris_model_f1_macro", "Macro F1 score of the deployed model"),
    }
    metrics["accuracy"].set(accuracy)
    metrics["f1"].set(f1)
    return metrics


saved = load_model()
model, features = saved["model"], saved["features"]
data = load_data()
metrics = start_metrics(saved["accuracy"], saved["f1"])


def predict(values, source):
    start = time.time()
    row = pd.DataFrame([values], columns=features)
    proba = model.predict_proba(row)[0]
    species = model.classes_[int(np.argmax(proba))]

    # Record the metrics for Prometheus
    metrics["latency"].observe(time.time() - start)
    metrics["predictions"].labels(species=species, source=source).inc()
    metrics["confidence"].observe(proba.max())
    for name, value in zip(features, values):
        metrics["feature"].labels(feature=name).set(value)
    return species, proba


st.title("🌸 Iris Species Classifier")
st.write("Move the sliders to a flower's measurements and the Random Forest model predicts "
         "its species: **setosa**, **versicolor** or **virginica**.")

# Inputs in the sidebar
with st.sidebar:
    st.header("Flower measurements (cm)")
    with st.form("inputs"):
        values = [st.slider(name.replace("_", " ").capitalize(), float(data[name].min()),
                            float(data[name].max()), float(round(data[name].mean(), 1)), step=0.1)
                  for name in features]
        submitted = st.form_submit_button("Predict species", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Simulate traffic")
    n_requests = st.number_input("Number of requests", min_value=5, max_value=500, value=25, step=5)
    simulate = st.button("Send simulated requests", use_container_width=True)

if submitted:
    st.session_state.last = predict(values, source="user")

left, right = st.columns([1, 1.3])
with left:
    st.subheader("Prediction")
    if "last" in st.session_state:
        species, proba = st.session_state.last
        st.success(f"Predicted species: **Iris {species}**")
        st.metric("Confidence", f"{proba.max():.1%}")
    else:
        st.info("Set the measurements in the sidebar and click **Predict species**.")
with right:
    if "last" in st.session_state:
        st.subheader("Class probabilities")
        st.bar_chart(pd.DataFrame({"species": model.classes_, "probability": st.session_state.last[1]}),
                     x="species", y="probability", height=260)

# Send a batch of noisy samples from the dataset, useful for filling the Grafana dashboard
if simulate:
    rng = np.random.default_rng()
    sample = data.sample(int(n_requests), replace=True)[features].to_numpy()
    results = [predict(list(np.clip(row + rng.normal(0, 0.2, 4), 0.1, None).round(1)), "simulated")[0]
               for row in sample]
    st.subheader("Simulated batch result")
    st.dataframe(pd.Series(results).value_counts().rename_axis("species").reset_index(name="count"),
                 hide_index=True)

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Model", "Random Forest")
c2.metric("Test accuracy", f"{saved['accuracy']:.1%}")
c3.metric("F1 (macro)", f"{saved['f1']:.3f}")
with st.expander("Training data: Iris dataset (150 flowers)"):
    st.scatter_chart(data, x="petal_length", y="petal_width", color="species", height=300)

st.caption("Arnav Mhatre · E002 · BTI DS Sem 11 · Batch E1 · Prometheus metrics on port 8001")
