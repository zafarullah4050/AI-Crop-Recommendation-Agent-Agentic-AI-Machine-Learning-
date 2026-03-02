from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import streamlit as st
import requests
import datetime

# -----------------------------
# STATE DEFINITION
# -----------------------------
class State(TypedDict):
    city: str
    season: str
    weather_summary: dict
    recommended_crops: list

# -----------------------------
# WEATHER TOOL (AGENT NODE)
# -----------------------------
def fetch_weather(state: State) -> State:
    city = state["city"]
    api_key = "5f48dbd07f484b619df34634260203"

    url = (
        "http://api.weatherapi.com/v1/forecast.json"
        f"?key={api_key}&q={city}&days=1"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        state["weather_summary"] = {
            "temperature_c": data["current"]["temp_c"],
            "humidity_percent": data["current"]["humidity"],
            "rainfall_mm": data["forecast"]["forecastday"][0]["day"]["totalprecip_mm"]
        }

    except Exception as e:
        state["weather_summary"] = {}
        st.error(f"Weather API error: {e}")

    return state

# -----------------------------
# SEASON ANALYSIS AGENT
# -----------------------------
def analyze_season(state: State) -> State:
    month = datetime.datetime.now().month

    if month in [10, 11, 12, 1, 2, 3]:
        season = "Rabi"
    elif month in [4, 5, 6, 7, 8, 9]:
        season = "Kharif"
    else:
        season = "Zaid"

    state["season"] = season
    return state

# -----------------------------
# CROP DECISION AGENT (RULE-BASED)
# -----------------------------
def recommend_crops(state: State) -> State:
    weather = state["weather_summary"]
    season = state["season"]

    if not weather:
        state["recommended_crops"] = []
        return state

    temp = weather["temperature_c"]
    humidity = weather["humidity_percent"]
    rainfall = weather["rainfall_mm"]

    crops = []

    # -------- RABI --------
    if season == "Rabi":
        if temp < 20:
            crops.extend(["Wheat", "Barley"])
        if humidity < 60:
            crops.append("Chickpea")
        crops.append("Mustard")

    # -------- KHARIF --------
    elif season == "Kharif":
        if rainfall > 80 and humidity > 60:
            crops.append("Rice")
        if temp > 30 and rainfall < 50:
            crops.append("Cotton")
        if 25 <= temp <= 35:
            crops.append("Maize")
        crops.append("Sugarcane")

    # -------- ZAID --------
    elif season == "Zaid":
        if temp > 30:
            crops.extend(["Watermelon", "Muskmelon"])
        crops.extend(["Cucumber", "Vegetables"])

    # Remove duplicates & limit output
    state["recommended_crops"] = list(dict.fromkeys(crops))[:5]
    return state

# -----------------------------
# LANGGRAPH SETUP
# -----------------------------
graph = StateGraph(State)

graph.add_node("fetch_weather", fetch_weather)
graph.add_node("analyze_season", analyze_season)
graph.add_node("recommend_crops", recommend_crops)

graph.add_edge(START, "fetch_weather")
graph.add_edge("fetch_weather", "analyze_season")
graph.add_edge("analyze_season", "recommend_crops")
graph.add_edge("recommend_crops", END)

app = graph.compile()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="AI Crop Recommendation Agent", layout="centered")

st.title("🌱 Agentic AI Crop Recommendation System (Pakistan)")
st.markdown(
    "Enter a city name. The AI agent will analyze **weather + season** "
    "and recommend suitable crops."
)

city = st.text_input("City Name", placeholder="e.g., Peshawar, Lahore, Karachi")

if st.button("Get Recommendations"):
    if city.strip():
        with st.spinner("Agent is analyzing climate and season..."):
            result = app.invoke({"city": city.strip()})

            if result["weather_summary"]:
                st.success("Decision generated successfully")

                st.subheader(f"📍 City: {result['city']}")
                st.subheader(f"🗓️ Season: {result['season']}")

                weather = result["weather_summary"]
                col1, col2, col3 = st.columns(3)
                col1.metric("🌡 Temperature (°C)", weather["temperature_c"])
                col2.metric("💧 Humidity (%)", weather["humidity_percent"])
                col3.metric("🌧 Rainfall (mm)", weather["rainfall_mm"])

                st.subheader("🌾 Recommended Crops")
                for crop in result["recommended_crops"]:
                    st.write("•", crop)
            else:
                st.error("Failed to generate recommendations.")
    else:
        st.error("Please enter a valid city name.")
