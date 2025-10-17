from fastapi import FastAPI, Request
import joblib
import numpy as np
import json
import os

app = FastAPI()

# ✅ Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Use relative paths (this works locally + on Render)
model_path = os.path.join(BASE_DIR, "stress_model.pkl")
features_path = os.path.join(BASE_DIR, "top_features.pkl")
encoders_path = os.path.join(BASE_DIR, "top_feature_encoders.pkl")

# ✅ Load once when server starts
model = joblib.load(model_path)
top_features = joblib.load(features_path)
top_feature_encoders = joblib.load(encoders_path)

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

import requests  # add this if not already imported

API_KEY = os.getenv("OPENROUTER_API_KEY")

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")

        system_prompt = {
            "role": "system",
            "content": """You are CalmViz, a friendly and empathetic stress-relief assistant.
            Respond warmly in 2–3 sentences. If asked to switch to Mandarin, reply:
            '你好！今天我可以帮你做些什么呢？'"""
        }

        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [system_prompt, {"role": "user", "content": user_message}]
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "X-Title": "CalmViz Chat"
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 json=payload, headers=headers)

        # Return the JSON reply directly
        return response.json()

    except Exception as e:
        return {"error": str(e)}
