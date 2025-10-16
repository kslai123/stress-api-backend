from fastapi import FastAPI, Request
import joblib
import numpy as np
import json

app = FastAPI()

# Load model once (reuse for all requests)
model = joblib.load(r"C:\xampp\htdocs\stress_prediction\stress_model.pkl")
top_features = joblib.load(r"C:\xampp\htdocs\stress_prediction\top_features.pkl")
top_feature_encoders = joblib.load(r"C:\xampp\htdocs\stress_prediction\top_feature_encoders.pkl")

@app.post("/predict")
async def predict(request: Request):
    try:
        raw_input_data = await request.json()

        input_values = []
        for feature_name in top_features:
            if feature_name not in raw_input_data:
                return {"error": f"Missing feature: {feature_name}"}
            raw_value = raw_input_data[feature_name]

            if feature_name in top_feature_encoders:
                encoder = top_feature_encoders[feature_name]
                encoded_val = encoder.transform([raw_value])[0]
                input_values.append(encoded_val)
            else:
                input_values.append(float(raw_value))

        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        activities = {
            "Low": "Listen to calming music",
            "Medium": "Go for a walk",
            "High": "Take deep breaths and meditate"
        }

        return {
            "stress_level": str(prediction),
            "recommendation": activities.get(prediction, "No suggestion available")
        }

    except Exception as e:
        return {"error": str(e)}
