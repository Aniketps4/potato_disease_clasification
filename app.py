import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ==============================
# ⚠️ MUST BE FIRST STREAMLIT COMMAND
# ==============================
st.set_page_config(page_title="Potato Disease Detection", page_icon="🥔", layout="wide")

# ==============================
# Load environment variables
# ==============================
load_dotenv()  # loads GOOGLE_API_KEY

# ==============================
# Directories & Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "potatoes.h5")

# ==============================
# App Title
# ==============================
st.title("🥔 AI Aloo Kisan Mitra / एआई आलू किसान मित्र")
st.markdown("Upload a potato leaf image to detect diseases and get expert advice powered by Gemini LLM.")
st.markdown("रोगों का पता लगाने और जेमिनी एलएलएम द्वारा संचालित विशेषज्ञ सलाह प्राप्त करने के लिए आलू के पत्ते की छवि अपलोड करें।")

# ==============================
# Load Potato Disease Model
# ==============================
@st.cache_resource
def load_potato_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_potato_model()

# Class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# ==============================
# Image Preprocessing
# ==============================
def preprocess_image(image: Image.Image):
    img = image.resize((256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# ==============================
# Prediction
# ==============================
def predict_disease(image: Image.Image):
    if model is None:
        return None, None
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = int(np.max(prediction[0]) * 100)
    return predicted_class, confidence

# ==============================
# Gemini LLM Setup
# ==============================
@st.cache_resource
def setup_llm_chain():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set your GOOGLE_API_KEY in the .env file.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash",  # Or gemini-pro
            google_api_key=api_key,
            temperature=0.1
        )

        prompt_template = """
        You are an expert plant pathologist specializing in potato diseases. 
        Based on the detected disease: {disease}, provide clear, actionable, concise advice to a farmer in a farmer-friendly format.

        STRICTLY structure your response EXACTLY as follows, with BOTH English and Hindi sections. Use bullet points, emojis (✅ for Preventive, 🛠 for Manage/Overcome, 🌱 for Care/Further Steps), and keep each section to 2-3 short sentences max. Be empathetic and simple.

        🧪 Predicted Disease: {disease}

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

        prompt = PromptTemplate(
            input_variables=["disease"],
            template=prompt_template
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        return chain

    except Exception as e:
        st.error(f"Error initializing Gemini LLM: {e}")
        return None

# ==============================
# Main App Logic
# ==============================
def main():
    chain = setup_llm_chain()
    if chain is None:
        st.warning("Gemini LLM is not configured. You can still upload an image for prediction.")
    
    uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting disease..."):
            predicted_class, confidence = predict_disease(image)

        if predicted_class:
            st.success(f"**Predicted Disease**: {predicted_class.replace('Potato___', '')}")
            st.info(f"**Confidence**: {confidence}%")

            if chain:
                with st.spinner("Generating expert advice..."):
                    try:
                        disease_name = predicted_class.replace('Potato___', '')
                        response = chain.run(disease=disease_name)
                        st.markdown("### 🩺 Advice from AI kisan mitra / AI किसान मित्र की सलाह")
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"Error generating advice: {e}")
        else:
            st.warning("Prediction failed. Please try another image.")

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload a clear image of a potato leaf.
    2. The model will classify it as Early Blight, Late Blight, or Healthy.
    3. Get tailored advice on management, prevention, and next steps.
    """)
    st.markdown("""
    1. आलू के पत्ते की स्पष्ट छवि अपलोड करें।
    2. मॉडल इसे प्रारंभिक झुलसा, विलंबित झुलसा या स्वस्थ के रूप में वर्गीकृत करेगा।
    3. प्रबंधन, रोकथाम और अगले कदमों पर अनुकूलित सलाह प्राप्त करें।
    """)
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("Trained on potato disease dataset with 3 classes.")
    st.markdown("Powered by TensorFlow & LangChain + Gemini.")

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    main()
