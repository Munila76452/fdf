import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor

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
def load_diagnosis_model(model_path):
    """
    Loads the pre-trained Keras model for disease diagnosis.
    """
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.error("Please ensure the 'plant_disease_model.h5' file is in the same directory as this script.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the diagnosis model: {e}")
        st.error("This may be due to a corrupted file or a version mismatch. Please ensure the model file is valid.")
        return None

@st.cache_resource
def train_forecasting_model(file_path):
    """
    Loads the dataset, preprocesses it, and trains a RandomForestRegressor model.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.drop('filename', axis=1)
        categorical_features = ['class_name', 'location', 'soil_status', 'weather']
        df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        X = df_encoded.drop('predicted_outbreaks_next_day', axis=1)
        y = df_encoded['predicted_outbreaks_next_day']
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        return model, X.columns
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return None, None
    except Exception as e:
        st.error(f"Error training forecasting model: {e}")
        return None, None

def predict_image(model, image_to_predict):
    """
    Takes a loaded Keras model and a PIL image, preprocesses the image,
    and returns the predicted class and confidence score.
    """
    try:
        img = image_to_predict.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)
        return predicted_class, confidence
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None

def predict_outbreak(model, training_columns, input_data):
    """
    Makes a prediction for a single new data point.
    """
    try:
        new_df = pd.DataFrame([input_data])
        new_df_encoded = pd.get_dummies(new_df, drop_first=True)
        new_df_aligned = new_df_encoded.reindex(columns=training_columns, fill_value=0)
        prediction = model.predict(new_df_aligned)
        return prediction[0]
    except Exception as e:
        st.error(f"Error making forecast: {e}")
        return None

def get_coordinates(place):
    """
    Geocodes a place name to latitude and longitude using Nominatim.
    """
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
    """
    Creates a Folium map with markers and optional heatmap.
    """
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
    model = load_diagnosis_model('plant_disease_model.h5')
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image for disease diagnosis."
    )

    if model is None:
        st.warning("The diagnosis model could not be loaded. Please check the file path and integrity.")
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.divider()
        if st.button("Diagnose Disease"):
            with st.spinner("🔍 Analyzing the image..."):
                predicted_class, confidence = predict_image(model, image)
            if predicted_class is not None:
                st.success("Analysis Complete!")
                st.session_state.predicted_disease = predicted_class
                formatted_class = predicted_class.replace('___', ' ').replace('_', ' ')
                if 'healthy' in formatted_class.lower():
                    st.markdown(f"### Diagnosis: <span style='color:green;'>**{formatted_class}**</span>", unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"### Diagnosis: <span style='color:red;'>**{formatted_class}**</span>", unsafe_allow_html=True)
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
    else:
        st.info("Please upload an image to begin the diagnosis.")

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

    # Load and train forecasting model
    forecasting_model, train_cols = train_forecasting_model('final_combined_dataset.csv')

    if forecasting_model is None or train_cols is None:
        st.warning("The forecasting model could not be loaded. Please ensure 'final_combined_dataset.csv' is in the same directory.")
    else:
        # Get unique locations and other options from the dataset
        try:
            df = pd.read_csv('final_combined_dataset.csv')
            locations = sorted(df['location'].unique())
            soil_statuses = sorted(df['soil_status'].unique())
            weathers = sorted(df['weather'].unique())
        except FileNotFoundError:
            st.error("Dataset 'final_combined_dataset.csv' not found. Please upload the file.")
            locations = ['Mumbai', 'Pune', 'Nashik', 'Satara', 'Sangli', 'Kolhapur', 'Amravati', 'Aurangabad', 'Solapur', 'Nagpur']
            soil_statuses = ['Dry', 'Moist', 'Wet', 'Cracked', 'Waterlogged']
            weathers = ['Sunny', 'Cloudy', 'Rainy', 'Foggy', 'Stormy']

        # Input form for forecasting
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
                    predicted_outbreaks = predict_outbreak(forecasting_model, train_cols, input_data)
                    if predicted_outbreaks is not None:
                        st.success("Forecast Generated!")
                        st.metric(label="Predicted Outbreaks (Next Day)", value=f"{predicted_outbreaks:.2f}")
                        # Display map with the forecasted location
                        forecast_data = [{
                            'lat': latitude,
                            'long': longitude,
                            'class': disease,
                            'severity': predicted_outbreaks / 20.0  # Normalize for heatmap (assuming max outbreaks ~20)
                        }]
                        st.subheader("Forecast Location")
                        folium_map = create_map(forecast_data, center=[latitude, longitude], zoom=10, heatmap=False)
                        if folium_map:
                            st_folium(folium_map, width=700, height=500)
                    else:
                        st.error("Failed to generate forecast. Please check your inputs.")
