import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import pandas as pd
import time

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Plant Disease Diagnosis & Outbreak Visualization",
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
def load_model(model_path):
    """
    Loads the pre-trained Keras model from the specified H5 file.
    """
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.error("Please ensure the 'plant_disease_model.h5' file is in the same directory as this script.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.error("This may be due to a corrupted file or a version mismatch. Please ensure the model file is valid.")
        return None

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

def get_coordinates(place):
    """
    Geocodes a place name to latitude and longitude using Nominatim.
    """
    geolocator = Nominatim(user_agent="plant_disease_tracker")
    try:
        location = geolocator.geocode(place + ", India")
        time.sleep(1)  # Avoid rate limit
        if location:
            return location.latitude, location.longitude
        else:
            st.error(f"Could not find coordinates for {place}")
            return None, None
    except Exception as e:
        st.error(f"Error geocoding {place}: {e}")
        return None, None

def create_map(data):
    """
    Creates a Folium map with markers and heatmap based on outbreak data.
    """
    try:
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Center on India
        heat_data = [[row['lat'], row['long'], row['severity']] for row in data if pd.notnull(row['lat']) and pd.notnull(row['long'])]
        if heat_data:
            HeatMap(heat_data, radius=15).add_to(m)
        for row in data:
            if pd.notnull(row['lat']) and pd.notnull(row['long']):
                color = 'green' if 'healthy' in row['class'].lower() else 'red'
                folium.Marker(
                    [row['lat'], row['long']],
                    popup=f"{row['class'].replace('___', ' ').replace('_', ' ')}: {row['severity']:.2f}",
                    icon=folium.Icon(color=color)
                ).add_to(m)
        return m
    except Exception as e:
        st.error(f"Error generating map: {e}")
        return None

# --- 4. STREAMLIT UI ---

st.title("🌿 Plant Disease Diagnosis & Outbreak Visualization")
st.markdown("Upload a plant leaf image for disease diagnosis or report a disease outbreak to visualize on a map.")

# Initialize session state for outbreak data
if 'outbreak_data' not in st.session_state:
    st.session_state.outbreak_data = []

# Tabs for Diagnosis and Outbreak Visualization
tab1, tab2 = st.tabs(["Disease Diagnosis", "Outbreak Visualization"])

# --- Disease Diagnosis Tab ---
with tab1:
    st.header("Disease Diagnosis")
    model = load_model('plant_disease_model.h5')
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image for disease diagnosis."
    )

    if model is None:
        st.warning("The model could not be loaded. Please check the file path and integrity.")
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.divider()
        if st.button("Diagnose Disease"):
            with st.spinner("🔍 Analyzing the image..."):
                predicted_class, confidence = predict_image(model, image)
            if predicted_class is not None:
                st.success("Analysis Complete!")
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
    st.markdown("Report a plant disease outbreak by specifying the location, disease, and severity.")

    # Input form
    with st.form(key="outbreak_form"):
        place = st.text_input("Location (e.g., Mumbai, Maharashtra)", value="Mumbai, Maharashtra")
        disease = st.selectbox("Disease", options=CLASS_NAMES)
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
                    st.success(f"Added {disease} at {place} to the outbreak map.")
                else:
                    st.error("Unable to add location to the map. Please try a different location.")

    # Display the outbreak map
    if st.session_state.outbreak_data:
        st.subheader("Outbreak Map")
        folium_map = create_map(st.session_state.outbreak_data)
        if folium_map:
            st_folium(folium_map, width=700, height=500)
    else:
        st.info("No outbreak data available. Add data using the form above to visualize the map.")
