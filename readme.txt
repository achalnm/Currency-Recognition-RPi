Handheld Currency Detection Device for the Visually Impaired

Overview:
This project provides a portable, offline AI-powered currency recognition system designed for visually impaired individuals. It uses computer vision and deep learning to recognize Indian currency notes in real time, without requiring a smartphone or internet connection.

The AI model runs efficiently on Raspberry Pi (or similar hardware) and provides instant audio feedback to identify currency notes.

Team Members:
Achal N
Aruna A Shenoy
Bhushan M V
Pujitha D R

Project Files:
- trainh5.py: Train the CNN model on your dataset; saves both .h5 and .tflite models
- test_single_image.py: Test the trained .h5 model on a single image
- raspi.py: Capture images from a webcam on Raspberry Pi and perform real-time inference using .tflite model
- convert_to_tflite.py: Optional script to convert .h5 model to .tflite
- requirements.txt: Python dependencies
- docs/FYP_Report.pdf: Final year project report (optional but recommended)

Hardware Requirements:
- Raspberry Pi 3B+ or newer
- USB Webcam / Pi Camera (minimum 480p)
- Battery Pack for portability
- Speaker for audio feedback
- Optional: Push button to trigger currency recognition

Software Requirements:
- Python 3.8+
- TensorFlow / Keras
- tflite-runtime
- OpenCV
- pyttsx3
- Pillow (PIL)

Dataset Structure:
Organize your currency dataset like this:
dataset/
├── 10/
├── 20/
├── 50/
├── 100/
├── 200/
├── 500/
└── others/
Each folder should contain images of that denomination.

Usage Instructions:

1. Train the Model:
- Update DATASET_DIR and model save paths in trainh5.py
- Run: python trainh5.py
- Saves .h5 and .tflite models

2. Test on a Single Image:
- Update MODEL_PATH and IMG_PATH in test_single_image.py
- Run: python test_single_image.py

3. Deploy on Raspberry Pi:
- Copy the .tflite model to the Pi
- Update MODEL_PATH in raspi.py
- Install dependencies: pip install -r requirements.txt
- Run: python3 raspi.py
- The script will capture an image, classify it, and provide audio feedback

Acknowledgements:
This project provides an offline, portable, and cost-effective solution for visually impaired individuals to confidently identify Indian currency using a lightweight MobileNetV2 CNN model.
