from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

# TensorFlow Serving endpoint
endpoint = "http://localhost:8051/v1/models/potatoes_di:predict"

# Classes
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello my name is Aniket"}

def read_file_as_image(data) -> np.ndarray:
    # Resize to model input size
    image = Image.open(BytesIO(data)).resize((224, 224))
    image = np.array(image) / 255.0  # normalize if trained with normalization
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file
    data = await file.read()
    image = read_file_as_image(data)

    # Add batch dimension
    img_batch = np.expand_dims(image, 0)

    # Prepare request
    json_data = {"instances": img_batch.tolist()}

    # Send to TensorFlow Serving
    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    # Interpret result
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "class": predicted_class,
        "confidence": confidence
    }
