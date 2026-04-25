from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model + scaler
with open("../model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Credit Risk API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to array
        features = np.array(list(data.values())).reshape(1, -1)

        # Scale
        features = scaler.transform(features)

        # Predict
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        return {
            "prediction": int(pred),
            "risk": "High Risk" if pred == 1 else "Low Risk",
            "probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}