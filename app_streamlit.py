import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Plant-AI Diagnosis",
    page_icon="🌿",
    layout="wide"
)

# --- 2. STYLING ---
# Inject custom CSS for a better UI
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1.5rem;
    }
    /* Title styling */
    h1 {
        color: #38761d;
        text-align: center;
    }
    /* Result card styling */
    .result-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .result-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    /* Diagnosis text styling */
    .diagnosis-healthy {
        color: #2e7d32;
        font-size: 24px;
        font-weight: bold;
    }
    .diagnosis-diseased {
        color: #c62828;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. MODEL AND CLASS NAMES ---
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scor'
    'h', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- 4. HELPER FUNCTIONS (with Caching) ---
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained Keras model and caches it."""
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file not found at {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading the model: {e}")
        return None

def predict_image(model, image_to_predict):
    """Preprocesses the image and returns the prediction."""
    try:
        # This model expects 256x256 images
        img = image_to_predict.resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # This model includes a Rescaling layer, so no manual normalization is needed
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)
        return predicted_class, confidence
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None

# --- 5. UI LAYOUT ---

# --- Sidebar ---
with st.sidebar:
    st.title("🌿 Plant-AI Controls")
    st.markdown("---")
    
    # Load the model
    model = load_model('best_model.keras')
    
    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )
    
    with st.expander("About this App"):
        st.info("""
            This application uses a deep learning model to diagnose diseases in plant leaves from images. 
            
            **How to use:**
            1.  Upload an image using the uploader above.
            2.  The image will be displayed on the right.
            3.  The AI will automatically analyze it and show the diagnosis.
        """)

# --- Main Page ---
st.title("Plant Disease Diagnosis")
st.markdown("---")

if uploaded_file is None:
    st.info("Please upload an image using the sidebar to get started.")
elif model is None:
    st.warning("Model not loaded. Please check the sidebar for error messages.")
else:
    # Create two columns for image and results
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Analyzing the image..."):
            predicted_class, confidence = predict_image(model, image)
        
        if predicted_class is not None:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            st.subheader("Diagnosis Result")
            
            formatted_class = predicted_class.replace('___', ' ').replace('_', ' ')
            
            if 'healthy' in formatted_class.lower():
                st.markdown(f'<p class="diagnosis-healthy">{formatted_class}</p>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<p class="diagnosis-diseased">{formatted_class}</p>', unsafe_allow_html=True)
            
            st.subheader("Confidence Level")
            st.progress(int(confidence))
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)

