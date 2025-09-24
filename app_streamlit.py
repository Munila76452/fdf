import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Plant Disease Diagnosis",
    page_icon="🌿",
    layout="centered"
)

# --- 2. MODEL AND CLASS NAMES ---
# This list must be in the exact same order as the classes your model was trained on.
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

# --- 3. HELPER FUNCTIONS (with Caching) ---

@st.cache_resource
def load_model(model_path):
    """
    Loads the pre-trained Keras model from the specified file.
    Uses caching to avoid reloading the model on every interaction.
    """
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.error("Please ensure the 'best_model.keras' file is in the same directory as this script.")
        return None
    try:
        # Load the model from the .keras format
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def predict_image(model, image_to_predict):
    """
    Takes a loaded Keras model and a PIL image, preprocesses the image,
    and returns the predicted class and confidence score.
    """
    try:
        # This model expects images of size 256x256
        img = image_to_predict.resize((256, 256))
        
        # Convert the image to a NumPy array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # This model has a Rescaling layer, so no manual normalization is needed.
        # Create a batch of 1 for the model input
        img_array = tf.expand_dims(img_array, 0)

        # Make prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Get the predicted class and confidence
        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        return predicted_class, confidence
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None

# --- 4. STREAMLIT UI ---

st.title("🌿 Plant Disease Diagnosis")
st.markdown("Upload an image of a plant leaf, and the AI will identify the disease.")

# Load the model
model = load_model('best_model.keras')

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a plant leaf image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image for disease diagnosis."
)

if model is None:
    st.warning("The model could not be loaded. Please check the file path and integrity.")
elif uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.divider()
    
    # Predict and display the result when the button is clicked
    if st.button("Diagnose Disease"):
        with st.spinner("🔍 Analyzing the image..."):
            predicted_class, confidence = predict_image(model, image)
        
        if predicted_class is not None:
            st.success("Analysis Complete!")
            
            # Format the output for better readability
            formatted_class = predicted_class.replace('___', ' ').replace('_', ' ')
            
            # Display the result with color coding
            if 'healthy' in formatted_class.lower():
                st.markdown(f"### Diagnosis: <span style='color:green;'>**{formatted_class}**</span>", unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"### Diagnosis: <span style='color:red;'>**{formatted_class}**</span>", unsafe_allow_html=True)
            
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
else:
    st.info("Please upload an image to begin the diagnosis.")
