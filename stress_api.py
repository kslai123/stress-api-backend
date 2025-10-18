from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import json
import os
import requests

app = FastAPI()

# ✅ Allow frontend access from any domain (or restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ✅ Get OpenRouter API Key
API_KEY = os.getenv("OPENROUTER_API_KEY")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "CalmViz API is running",
        "endpoints": ["/predict", "/chat", "/suggest"]
    }

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

@app.post("/suggest")
async def suggest(request: Request):
    """
    Accepts JSON:
      { "activity": "...", "budget": "...", "lat": 1.23, "lng": 103.45, "town": "Kuala Lumpur" }

    Returns: the assistant-provided JSON object (parsed) or an error object.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    activity = body.get("activity", "") or ""
    budget = body.get("budget", "") or ""
    lat = body.get("lat", None)
    lng = body.get("lng", None)
    town = body.get("town", None)

    # Build location part exactly like PHP
    locationPart = ""
    if lat and lng:
        locationPart += f"User location: latitude = {lat}, longitude = {lng}. "
    if town:
        locationPart += f"User town: {town}."

    # Full RULES / JSON format prompt (exact same as PHP)
    RULES_FORMAT = '''
RESPONSE FORMAT: Return your answer ONLY as a JSON object.  
Always follow this exact structure:

{
  "action_title": "A very short, catchy name for the action (3–5 words max)",
  "action_description": "One clear sentence explaining the action and why it helps.",
  "action_steps": [
    "Step 1...",
    "Step 2...",
    "Step 3..."
  ],
  "extras": {
    "timing": "",            // Only if activity = "breathing", else ""
    "duration": "",          // Only if activity = "exercise", else ""
  },
  "links": {
    "spotify": "",           // Only if activity = "music", else ""
    "youtube": "",           // Only if activity = "music", else ""
    "video": "",             // Only if activity = "exercise" or activity = "watching videos", else ""
    "reading": "",           // Only if activity = "reading", else ""
    "game": "https://www.crazygames.com/t/relaxing" // Only if activity = "game", else ""
  },
  "travel": {
    "title": "",               // If budget exists → descriptive trip name. If no budget → leave empty.
    "budget": "",              // Copy user's budget. If empty → no travel plan should be generated.
    "plan": [],                // Must remain [] if budget is empty
    "budget_breakdown": {}     // Must remain {} if budget is empty
  },
  "meals": {
    "nearby_food": [],        
    "restaurant_details": []  // Each item = { "name": "", "address": "", "dish": "", "price_range": "", "map_link": "" }
  }
}

RULES:
- If activity is **sleeping** → 
   * Fill `extras.duration` with recommended hours (high stress = Try to sleep more (about 8–9 hours).;, moderate = Aim for 7–8 hours of sleep., low = Maintain about 7 hours of sleep.).
   * In `action_steps`, include at least:
       1. Environment setup (temperature around 18–20°C, dark, quiet room).
       2. Pre-sleep routine (reading, meditation, stretching).
       3. Techniques to reduce stress (deep breathing, journaling, body scan).
       4. Consistent sleep schedule (same bedtime/wake time daily).
    * No need fill in links.

- If activity is **music** →
   * Always fill `links.spotify` with a Spotify search query for calming/relaxing music.
   * Always fill `links.youtube` with a YouTube search query for calming/relaxing music.
   * Do NOT use direct links to single videos/playlists (to avoid broken links).

- If activity is **breathing** → fill `extras.timing` (e.g., "4-4-4-4").
- If activity is **exercise** → fill `links.video` and mention duration in `extras.duration`, and also inside `action_steps`.
- If activity is **traveling** →
   * If budget is missing or empty → 
       - Set "travel.title" = ""  
       - Set "travel.plan" = []  
       - Set "travel.budget_breakdown" = {}  
   * Only if budget exists → then generate travel.title, plan, and budget_breakdown.
   * The user will not provide a destination, so you must select a suitable, budget-friendly destination in Malaysia.
   * Always include a `travel.title` for the trip (e.g., "Exploring Penang" or "Malacca Cultural Getaway").
   * Always copy the user's budget into `travel.budget`.
   * Always include `travel.title` with a descriptive destination name. 
   *If user did not give a place, choose a suitable student-friendly travel destination in Malaysia
   * In `travel.plan`, create a **detailed multi-day travel itinerary** (2–4 days depending on budget).
   * Each day must include:
       - **Title**: A short title (e.g., "Day 1: Arrival & Heritage Walk").
       - **Morning**: Main activity with location name and short description.
       - **Afternoon**: Main activity with location name and short description.
       - **Evening**: Main activity with location name and short description.
       - **Meals**: Suggested food with dish names and recommended restaurants/stalls.
       - **Accommodation**: Hotel/hostel name, type, location, and approx. cost per night.
   * Include a `budget_breakdown` object inside `travel`:
       - Transport (bus/flight/taxi costs)
       - Accommodation (per night × nights)
       - Meals (estimated daily food cost)
       - Activities/Attractions (tickets, tours, etc.)
       - Miscellaneous (souvenirs, snacks, local transport)
       - Total (sum of all categories, must not exceed budget)
   * Mention **specific locations/attractions** (e.g., "Penang Hill", "Batu Ferringhi Beach") instead of general words like "temple" or "beach".
   * Ensure the flow looks like a **professional travel agency itinerary**, clear and structured.
   * Keep the plan **student-friendly, affordable, and stress-relieving** (focus on budget-friendly food, hostels, and free/low-cost attractions).

- If activity is **watching videos** → fill `links.video`.
   * Always fill `links.video` with a YouTube search query for funny or stress-relief videos.

- If activity is reading → fill links.reading.
   * Always fill links.reading with a free eBook or article link (e.g., Project Gutenberg, Open Library, or Google Books).
   * Choose stress-relief, motivational, or light novels suitable for relaxation.
   * Ensure the link is accessible and not behind a paywall.
   
- If activity is **enjoying meals** →  
   * Use provided `lat` and `lng` (user's location) to recommend **3–5 nearby restaurants or street food stalls**.  
   * Fill `meals.nearby_food` with short names of suggested dishes (e.g., "Nasi Lemak", "Char Kway Teow").  
   * Fill `meals.restaurant_details` with objects like:  
     {
       "name": "Ali Nasi Lemak",
       "address": "123 Jalan Ampang, Kuala Lumpur",
       "dish": "Nasi Lemak with fried chicken",
       "price_range": "RM8–RM12"
     }  
   * Keep suggestions **budget-friendly and student-friendly**.  
   * If location cannot be used → return empty `meals` fields.  
   * For each restaurant in meals.restaurant_details, always include "map_link" as a Google Maps link using the exact restaurant name + full address (if available). 
- If an exact link cannot be generated, use a Google Maps search query with the restaurant name + town.

- If activity is not relevant for a field → leave it empty (`""` or `[]`).
- Do NOT add extra fields outside this schema.
- Always return **valid JSON** only, with no text before or after.
'''

    prompt = f"You are an AI assistant that recommends stress-relief activities.\nUser input activity: {activity}\nUser budget: {budget}\n{locationPart}\n{RULES_FORMAT}"

    # Build OpenRouter request (exactly like PHP)
    payload = {
        "model": "openai/gpt-5-pro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that only outputs valid JSON."},
            {"role": "user", "content": prompt}
        ]
    }

    # Make API call to OpenRouter
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        response_data = response.json()
        
        # Extract AI message content (same as PHP logic)
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            # Parse the JSON string from AI and return it
            return json.loads(content)
        else:
            raise HTTPException(status_code=500, detail="No response from AI")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)