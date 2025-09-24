import streamlit as st
import numpy as np
import os
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import pandas as pd
import time

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Plant Disease Diagnosis & Outbreak Analysis",
    page_icon="🌿",
    layout="centered"
)

# --- 2. MODEL AND CLASS NAMES ---
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- 3. HELPER FUNCTIONS ---

@st.cache_resource
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure 'final_combined_dataset.csv' is uploaded.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def manual_forecast_outbreak(df, input_data):
    try:
        disease = input_data['class_name']
        location = input_data['location'].strip()
        temp = input_data['temperature']
        hum = input_data['humidity']
        soil = input_data['soil_status']
        weather = input_data['weather']

        mask = (
            (df['class_name'] == disease) &
            (df['location'] == location) &
            (df['temperature'].between(temp - 2, temp + 2)) &
            (df['humidity'].between(hum - 5, hum + 5)) &
            (df['soil_status'] == soil) &
            (df['weather'] == weather)
        )
        matching_data = df[mask]

        if matching_data.empty:
            fallback_data = df[df['class_name'] == disease]
            if fallback_data.empty:
                return df['predicted_outbreaks_next_day'].mean() if not df['predicted_outbreaks_next_day'].empty else 1.0
            return fallback_data['predicted_outbreaks_next_day'].mean()
        
        return matching_data['predicted_outbreaks_next_day'].mean()
    except Exception as e:
        st.error(f"Error in manual forecast: {e}")
        return None

def get_coordinates(place):
    geolocator = Nominatim(user_agent="plant_disease_tracker")
    try:
        location = geolocator.geocode(place + ", India")
        time.sleep(1)
        if location:
            return location.latitude, location.longitude
        else:
            st.error(f"Could not find coordinates for {place}")
            return None, None
    except Exception as e:
        st.error(f"Error geocoding {place}: {e}")
        return None, None

def create_map(data, center=[20.5937, 78.9629], zoom=5, heatmap=True):
    try:
        m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
        if heatmap:
            heat_data = [[row['lat'], row['long'], row['severity']] for row in data if pd.notnull(row['lat']) and pd.notnull(row['long'])]
            if heat_data:
                HeatMap(heat_data, radius=15).add_to(m)
        for row in data:
            if pd.notnull(row['lat']) and pd.notnull(row['long']):
                color = 'green' if 'healthy' in row['class'].lower() else 'red'
                popup_text = f"{row['class'].replace('___', ' ').replace('_', ' ')}: {row['severity']:.2f}"
                folium.Marker(
                    [row['lat'], row['long']],
                    popup=popup_text,
                    icon=folium.Icon(color=color)
                ).add_to(m)
        return m
    except Exception as e:
        st.error(f"Error generating map: {e}")
        return None

# --- 4. STREAMLIT UI ---

st.title("🌿 Plant Disease Diagnosis & Outbreak Analysis")
st.markdown("Diagnose plant diseases, visualize outbreaks, or forecast disease spread based on climate conditions.")

# Initialize session state
if 'outbreak_data' not in st.session_state:
    st.session_state.outbreak_data = []
if 'predicted_disease' not in st.session_state:
    st.session_state.predicted_disease = None

# Tabs
tab1, tab2, tab3 = st.tabs(["Disease Diagnosis", "Outbreak Visualization", "Outbreak Forecast"])

# --- Disease Diagnosis Tab ---
with tab1:
    st.header("Disease Diagnosis")
    st.warning("Disease Diagnosis is temporarily disabled due to dependency issues. Please use the other tabs.")
    st.info("Upload a plant leaf image to diagnose diseases (requires TensorFlow, which is not currently installed).")

# --- Outbreak Visualization Tab ---
with tab2:
    st.header("Outbreak Visualization")
    st.markdown("Report a plant disease outbreak to visualize on an interactive map of India. Markers are red for diseases and green for healthy plants.")

    default_disease = st.session_state.predicted_disease if st.session_state.predicted_disease in CLASS_NAMES else CLASS_NAMES[0]
    with st.form(key="outbreak_form"):
        place = st.text_input("Location (e.g., Mumbai, Maharashtra)", value="Mumbai, Maharashtra")
        disease = st.selectbox("Disease", options=CLASS_NAMES, index=CLASS_NAMES.index(default_disease))
        severity = st.slider("Severity", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        submit_button = st.form_submit_button("Add to Outbreak Map")

        if submit_button:
            with st.spinner("Processing location..."):
                lat, long = get_coordinates(place)
                if lat and long:
                    st.session_state.outbreak_data.append({
                        'lat': lat,
                        'long': long,
                        'class': disease,
                        'severity': severity
                    })
                    st.success(f"Added {disease.replace('___', ' ').replace('_', ' ')} at {place} to the outbreak map.")
                else:
                    st.error("Unable to add location to the map. Please try a different location.")

    if st.session_state.outbreak_data:
        st.subheader("Outbreak Map")
        folium_map = create_map(st.session_state.outbreak_data)
        if folium_map:
            st_folium(folium_map, width=700, height=500)
    else:
        st.info("No outbreak data available. Add data using the form above to visualize the map.")

# --- Outbreak Forecast Tab ---
with tab3:
    st.header("Outbreak Forecast")
    st.markdown("Predict the number of disease outbreaks for the next day based on climate and location data.")

    df = load_dataset('final_combined_dataset.csv')

    if df is None:
        st.warning("The dataset could not be loaded. Please ensure 'final_combined_dataset.csv' is in the same directory.")
    else:
        try:
            locations = sorted(df['location'].unique())
            soil_statuses = sorted(df['soil_status'].unique())
            weathers = sorted(df['weather'].unique())
        except Exception as e:
            st.error(f"Error processing dataset: {e}")
            locations = ['Mumbai', 'Pune', 'Nashik', 'Satara', 'Sangli', 'Kolhapur', 'Amravati', 'Aurangabad', 'Solapur', 'Nagpur']
            soil_statuses = ['Dry', 'Moist', 'Wet', 'Cracked', 'Waterlogged']
            weathers = ['Sunny', 'Cloudy', 'Rainy', 'Foggy', 'Stormy']

        with st.form(key="forecast_form"):
            default_disease = st.session_state.predicted_disease if st.session_state.predicted_disease in CLASS_NAMES else CLASS_NAMES[0]
            disease = st.selectbox("Disease", options=CLASS_NAMES, index=CLASS_NAMES.index(default_disease), key="forecast_disease")
            location = st.selectbox("Location", options=locations, index=locations.index('Kolhapur') if 'Kolhapur' in locations else 0)
            latitude = st.number_input("Latitude", min_value=6.0, max_value=37.0, value=16.804442, step=0.000001, format="%.6f")
            longitude = st.number_input("Longitude", min_value=68.0, max_value=98.0, value=74.225031, step=0.000001, format="%.6f")
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=27.7, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=71.5, step=0.1)
            soil_status = st.selectbox("Soil Status", options=soil_statuses, index=soil_statuses.index('Moist') if 'Moist' in soil_statuses else 0)
            weather = st.selectbox("Weather", options=weathers, index=weathers.index('Foggy') if 'Foggy' in weathers else 0)
            submit_forecast = st.form_submit_button("Predict Outbreak")

            if submit_forecast:
                with st.spinner("Generating forecast..."):
                    input_data = {
                        'class_name': disease,
                        'location': location.strip(),
                        'latitude': latitude,
                        'longitude': longitude,
                        'temperature': temperature,
                        'humidity': humidity,
                        'soil_status': soil_status,
                        'weather': weather
                    }
                    predicted_outbreaks = manual_forecast_outbreak(df, input_data)
                    if predicted_outbreaks is not None:
                        st.success("Forecast Generated!")
                        st.metric(label="Predicted Outbreaks (Next Day)", value=f"{predicted_outbreaks:.2f}")
                        forecast_data = [{
                            'lat': latitude,
                            'long': longitude,
                            'class': disease,
                            'severity': predicted_outbreaks / 20.0
                        }]
                        st.subheader("Forecast Location")
                        folium_map = create_map(forecast_data, center=[latitude, longitude], zoom=10, heatmap=False)
                        if folium_map:
                            st_folium(folium_map, width=700, height=500)
                    else:
                        st.error("Failed to generate forecast. Please check your inputs.")
