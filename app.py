import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv  # Load .env file
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(page_title="Potato Disease Detection", page_icon="🥔", layout="wide")

# Title
st.title("🥔 AI Aloo Kisan Mitra / एआई आलू किसान मित्र")
st.markdown("Upload a potato leaf image to detect diseases and get expert advice powered by Gemini LLM.")
st.markdown("रोगों का पता लगाने और जेमिनी एलएलएम द्वारा संचालित विशेषज्ञ सलाह प्राप्त करने के लिए आलू के पत्ते की छवि अपलोड करें।")

# Load the model (using the full path provided)
@st.cache_resource
def load_potato_model():
    try:
        MODEL_PATH = os.path.join(BASE_DIR, "models", "potatoes.h5")

        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

model = load_potato_model()

# Class names from the training notebook
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Image preprocessing function
def preprocess_image(image):
    img = image.resize((256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Prediction function
def predict_disease(image):
    if model is None:
        return None, None
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = int(np.max(prediction[0]) * 100)
    return predicted_class, confidence

# Set up Gemini LLM with LangChain
@st.cache_resource
def setup_llm_chain():
    # Get API key from environment variable (loaded from .env)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set your GOOGLE_API_KEY in the .env file.")
        return None
    
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",  # Or use "gemini-pro" if preferred
        google_api_key=api_key,
        temperature=0.1  # Low temperature for factual responses
    )
    
    # Enhanced Prompt template for bilingual, concise output with emojis
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

# Main app logic
def main():
    chain = setup_llm_chain()
    if chain is None:
        return

    uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict
        with st.spinner("Detecting disease..."):
            predicted_class, confidence = predict_disease(image)

        if predicted_class:
            st.success(f"**Predicted Disease**: {predicted_class.replace('Potato___', '')}")
            st.info(f"**Confidence**: {confidence}%")

            # Generate advice with LLM
            with st.spinner("Generating expert advice..."):
                try:
                    disease_name = predicted_class.replace('Potato___', '')
                    response = chain.run(disease=disease_name)
                    st.markdown("### 🩺Advice from AI kisan mitra / AI किसान मित्र की सलाह")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error generating advice: {e}")
        else:
            st.warning("Prediction failed. Please try another image.")

if __name__ == "__main__":
    main()

# Sidebar for instructions
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
