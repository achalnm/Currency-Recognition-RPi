import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import pyttsx3
import time
import os
from datetime import datetime

class_labels = [
    "10 Rupees",   # class 0
    "100 Rupees",  # class 1
    "20 Rupees",   # class 2
    "200 Rupees",  # class 3
    "50 Rupees",   # class 4
    "500 Rupees",  # class 5
    "Others"       # class 6
]

def capture_image(save_folder="captured_images"):
    """Capture image from webcam and save to folder"""
    os.makedirs(save_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot access camera")

    time.sleep(2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture image")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"currency_{timestamp}.jpg"
    filepath = os.path.join(save_folder, filename)
    cv2.imwrite(filepath, frame)
    print(f"📷 Image saved to: {filepath}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def load_tflite_model(model_path):
    """Load TFLite model and allocate tensors"""
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_array, input_size=(224, 224)):
    """Preprocess captured image for model inference"""
    img = Image.fromarray(image_array).resize(input_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def infer(interpreter, input_data):
    """Run inference on prepared input data"""
    input_details = interpreter.get_input_details()[0]

    if input_details['dtype'] == np.uint8:
        input_data = (input_data * 255).astype(np.uint8)

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    return interpreter.get_tensor(output_details['index'])[0]

def speak_currency(label):
    """Convert label to speech output"""
    engine = pyttsx3.init()
    if label == "Others":
        engine.say("Unknown currency detected")
    else:
        currency_value = label.split()[0]
        engine.say(f"This is a {currency_value} rupee note. Thank you.")
    engine.runAndWait()

if __name__ == "__main__":
    MODEL_PATH = "<ADD_TFLITE_MODEL_PATH>/currency_model.tflite"
    IMAGE_FOLDER = "captured_images"

    try:
        print("Capturing image...")
        captured_image = capture_image(IMAGE_FOLDER)

        interpreter = load_tflite_model(MODEL_PATH)
        input_data = preprocess_image(captured_image)
        probabilities = infer(interpreter, input_data)

        predicted_class = np.argmax(probabilities)
        confidence = np.max(probabilities)
        label = class_labels[predicted_class]

        print(f"\n✅ Predicted currency: {label}")
        print(f"🔎 Confidence: {confidence:.2%}")
        speak_currency(label)

    except Exception as e:
        print(f"❌ Error: {str(e)}")


"""

Project: Handheld Currency Detection Device for the Visually Impaired

This script is part of a project to recognize Indian currency notes offline.
Refer to README.md for dataset structure, training instructions, and deployment on Raspberry Pi.

Team Members:
- 1JT21CS003 Achal N
- 1JT21CS019 Aruna A Shenoy
- 1JT21CS030 Bhushan M V
- 1JT21CS128 Pujitha D R

"""
