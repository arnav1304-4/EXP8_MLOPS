# train.py - trains a Random Forest that predicts the species of an Iris flower
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 1. Load the Iris dataset (150 flowers, 50 of each species)
df = pd.read_csv("data/iris.csv")
print("Rows:", len(df))
print(df["species"].value_counts().to_string())

# 2. Inputs are the four measurements in cm, target is the species
X = df[FEATURES]
y = df["species"]

# 3. Split into 80% train and 20% test, keeping the classes balanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train the model (shallow trees so it does not overfit 120 rows)
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate on the unseen test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
print("Test accuracy:", round(accuracy, 4))
print("Macro F1     :", round(f1, 4))
print(classification_report(y_test, y_pred))

# 6. Save the model with everything the app needs to use it
joblib.dump({"model": model, "features": FEATURES, "classes": list(model.classes_),
             "accuracy": accuracy, "f1": f1}, "model.pkl")
print("Saved model.pkl")
