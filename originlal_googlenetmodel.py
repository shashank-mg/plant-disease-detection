from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


@app.get("/ping")
async def ping():
    return "Hello"

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL = tf.keras.models.load_model('../model2.h5')

CLASS_NAMES = ['Coffee Healthy',
               'Cotton Healthy',
               'Cotton Leaf Blight',
               'Maize Blight',
               'Maize Common Rust',
               'Maize Gray Leaf Spot',
               'Maize Healthy',
               'Rice Brown Spot',
               'Rice Healthy',
               'Rice Hispa',
               'Rice LeafBlast',
               'Sugarcane Bacterial Blight',
               'Sugarcane Healthy',
               'Sugarcane Red Rot',
               'Coffee Miner',
               'Coffee Rust']


def read_file_as_image(img):
    image = np.array(Image.open(BytesIO(img)))
    return image


@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):

    bytes = await file.read()
    image = read_file_as_image(bytes)
    img_batch = np.expand_dims(image, 0).astype(np.float32)
    prediction = MODEL.predict(img_batch)
    return {CLASS_NAMES[np.argmax(prediction[0])], round(np.max(prediction[0])*100)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
