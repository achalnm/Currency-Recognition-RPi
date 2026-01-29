Handheld Currency Detection Device for the Visually Impaired

Team Members:
- Achal Nanjundamurthy
- Aruna A Shenoy
- Bhushan M V
- Pujitha D R

Overview:
This project provides a portable, offline AI-powered currency recognition system designed for visually impaired individuals. It uses computer vision and deep learning to recognize Indian currency notes in real time, without requiring a smartphone or internet connection.

The AI model runs efficiently on Raspberry Pi (or similar hardware) and provides instant audio feedback to identify currency notes.

Getting Started:
Follow the instructions below to set up and run the project on your PC or Raspberry Pi.

Project Files and Their Usage:
- trainh5.py: Train the CNN model on your dataset. Saves both .h5 and .tflite models. Update DATASET_DIR and MODEL_PATH before running.
- test_single_image.py: Test the trained .h5 model on a single image. Update MODEL_PATH and IMG_PATH before running.
- raspi.py: Deploy on Raspberry Pi for real-time currency detection using a webcam. Update MODEL_PATH to point to the .tflite model.
- convert_to_tflite.py: Optional script to convert a saved .h5 model to .tflite.
- requirements.txt: Python dependencies. Install via 'pip install -r requirements.txt'.

Note: Other files such as journals or certificates are for reference only and are not required to run the code.

Hardware Requirements:
- Raspberry Pi 3B+ or newer
- USB Webcam / Pi Camera (minimum 480p)
- Battery Pack for portability
- Speaker for audio feedback
- Optional: Push button to trigger currency recognition

Software Requirements:
- Python 3.8+ (Windows, Linux, or Raspberry Pi OS)
- TensorFlow / Keras
- tflite-runtime
- OpenCV
- pyttsx3
- Pillow (PIL)

Dataset Structure:
Each folder should contain images of that denomination. Images can be captured manually or downloaded from a dataset source. 
Organize your dataset like this:

dataset/
├── 10/
├── 20/
├── 50/
├── 100/
├── 200/
├── 500/
└── others/

Usage Instructions:

1. Train the Model:
- Ensure your dataset is ready and organized as above.
- Open trainh5.py and update:
  DATASET_DIR = "Add your dataset path here"
  MODEL_PATH = "Add path where model will be saved"
- Run:
  python trainh5.py
- The script will save:
  - .h5 model → used for testing on PC
  - .tflite model → used on Raspberry Pi

2. Test on a Single Image:
- Open test_single_image.py and update:
  MODEL_PATH = "Add path to saved .h5 model"
  IMG_PATH = "Add path to test image"
- Run:
  python test_single_image.py
- The predicted class will be printed in the console.

3. Deploy on Raspberry Pi:
- Copy the .tflite model to your Pi.
- Update raspi.py:
  MODEL_PATH = "Add path to .tflite model"
- Install dependencies:
  pip install -r requirements.txt
- Run:
  python3 raspi.py
- The script will capture an image, classify it in real-time, and provide audio feedback.

Acknowledgements:
This project provides an offline, portable, and cost-effective solution for visually impaired individuals to confidently identify Indian currency using a lightweight MobileNetV2 CNN model.
