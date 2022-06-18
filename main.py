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

interpreter_quant = tf.lite.Interpreter(
    model_path='../models/tmps/mnist_tflite_models1/mnist_model_quant1.tflite')
interpreter_quant.allocate_tensors()


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
    img_batch = np.expand_dims(image, axis=0).astype(np.float32)
    input_index = interpreter_quant.get_input_details()[0]['index']
    output_index = interpreter_quant.get_output_details()[0]["index"]
    interpreter_quant.set_tensor(input_index, img_batch)
    interpreter_quant.invoke()
    predictions = interpreter_quant.get_tensor(output_index)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
