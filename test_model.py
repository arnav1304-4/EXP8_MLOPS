# test_model.py - checks on the trained model, run locally and in GitHub Actions
import joblib
import pandas as pd

saved = joblib.load("model.pkl")


def predict(sepal_length, sepal_width, petal_length, petal_width):
    row = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                       columns=saved["features"])
    return saved["model"].predict(row)[0]


def test_model_file_has_parts():
    for key in ("model", "features", "classes", "accuracy", "f1"):
        assert key in saved


def test_scores_are_good():
    assert saved["accuracy"] >= 0.90
    assert saved["f1"] >= 0.90


def test_predicts_each_species():
    assert predict(5.1, 3.5, 1.4, 0.2) == "setosa"      # small petals
    assert predict(6.0, 2.9, 4.5, 1.5) == "versicolor"  # medium petals
    assert predict(6.9, 3.1, 5.8, 2.3) == "virginica"   # large petals


def test_probabilities_add_up_to_one():
    row = pd.DataFrame([[5.8, 2.7, 4.1, 1.0]], columns=saved["features"])
    assert abs(saved["model"].predict_proba(row).sum() - 1.0) < 1e-9
