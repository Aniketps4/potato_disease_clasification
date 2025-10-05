import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio

# ======================================
# 🌐 ENVIRONMENT SETUP
# ======================================
# Load .env file
load_dotenv()

# Handle async loop issues (important for Streamlit Cloud)
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Define BASE_DIR for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY is not set. LLM advice will not work.")

# ======================================
# 🧠 LOAD TENSORFLOW MODEL
# ======================================
@st.cache_resource
def load_potato_model():
    model_path = os.path.join(BASE_DIR, "models", "potatoes.h5")
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at {model_path}")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_potato_model()

# Class names must match your training labels
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# ======================================
# 🖼 IMAGE PREPROCESSING
# ======================================
def preprocess_image(image: Image.Image):
    img = image.resize((256, 256))  # match model input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ======================================
# 🔬 PREDICTION FUNCTION
# ======================================
def predict_disease(image: Image.Image):
    if model is None:
        return None, None
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = int(np.max(prediction[0]) * 100)
    return predicted_class, confidence

# ======================================
# 🧠 GEMINI RESPONSE
# ======================================
def generate_gemini_advice(disease_name: str):
    if not GOOGLE_API_KEY:
        return "⚠️ Gemini API key not configured."

    model_gemini = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    You are an expert plant pathologist specializing in potato diseases.
    Based on the detected disease: {disease_name}, provide clear, actionable, concise advice to a farmer in a farmer-friendly format.

    STRICTLY structure your response EXACTLY as follows, with BOTH English and Hindi sections.
    Use bullet points, emojis (✅ for Preventive, 🛠 for Manage/Overcome, 🌱 for Care/Further Steps),
    and keep each section to 2-3 short sentences max. Be empathetic and simple.

    🧪 Predicted Disease: {disease_name}

    🌿 Farmer Guidance (English + Hindi) 🌿

    **English:**
    ✅ **Preventive:** [1-2 concise bullet points on prevention]
    🛠 **Overcome:** [1-2 concise bullet points on immediate management]
    🌱 **Care:** [1-2 concise bullet points on monitoring/further steps]

    **हिन्दी (Hindi):**
    ✅ **रोकथाम:** [1-2 संक्षिप्त बुलेट पॉइंट्स रोकथाम पर]
    🛠 **समाधान:** [1-2 संक्षिप्त बुलेट पॉइंट्स तत्काल प्रबंधन पर]
    🌱 **देखभाल:** [1-2 संक्षिप्त बुलेट पॉइंट्स निगरानी/आगे के कदमों पर]

    Translate accurately to Hindi using Devanagari script. Do NOT add extra text or deviate from this format.
    """

    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating Gemini advice: {e}"

# ======================================
# 🟩 STREAMLIT UI
# ======================================
st.set_page_config(page_title="Potato Disease Detection", page_icon="🥔", layout="wide")

st.title("🥔 AI Aloo Kisan Mitra / एआई आलू किसान मित्र")
st.markdown("Upload a potato leaf image to detect diseases and get expert advice powered by Gemini LLM.")
st.markdown("रोगों का पता लगाने और जेमिनी एलएलएम द्वारा संचालित विशेषज्ञ सलाह प्राप्त करने के लिए आलू के पत्ते की छवि अपलोड करें।")

uploaded_file = st.file_uploader("📤 Upload Potato Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Detecting disease..."):
        predicted_class, confidence = predict_disease(image)

    if predicted_class:
        disease_name = predicted_class.replace("Potato___", "")
        st.success(f"**Predicted Disease:** {disease_name}")
        st.info(f"**Model Confidence:** {confidence}%")

        with st.spinner("🧠 Generating expert farmer advice..."):
            advice = generate_gemini_advice(disease_name)
            st.markdown("### 🩺 AI Kisan Mitra Advice / किसान मित्र की सलाह")
            st.markdown(advice)
    else:
        st.warning("⚠️ Prediction failed. Please try another image.")

# ======================================
# 📌 SIDEBAR
# ======================================
with st.sidebar:
    st.header("📌 Instructions")
    st.markdown("""
    1. Upload a clear potato leaf image.  
    2. The model classifies it as Early Blight, Late Blight, or Healthy.  
    3. Get AI-powered actionable advice in English + Hindi.
    """)
    st.markdown("""
    1. आलू के पत्ते की स्पष्ट छवि अपलोड करें।  
    2. मॉडल इसे प्रारंभिक झुलसा, विलंबित झुलसा या स्वस्थ के रूप में वर्गीकृत करेगा।  
    3. आपको हिंदी और अंग्रेजी में सलाह प्राप्त होगी।
    """)
    st.divider()
    st.caption("🧠 Powered by TensorFlow + Google Gemini 1.5 Flash")
