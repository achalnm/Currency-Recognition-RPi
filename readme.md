# Handheld Currency Detection Device for the Visually Impaired

## Project Overview
This project, developed at Jyothy Institute of Technology (2024-2025), provides a portable, offline AI-powered currency recognition system designed for visually impaired individuals. It utilizes computer vision and deep learning to identify Indian currency notes in real time without requiring a smartphone or internet connection.

## Technical Specifications
- **Architecture**: MobileNetV2 (Convolutional Neural Network)
- **Framework**: TensorFlow Lite (TFLite) for Edge Computing
- **Design Patterns**: Model-View-Controller (MVC) and Facade Pattern
- **Inference Engine**: TFLite Runtime for low-latency processing on ARM-based hardware
- **Image Preprocessing**: OpenCV-based resizing (224x224), normalization, and grayscale conversion (where applicable)

## Hardware Requirements
- Raspberry Pi 3B+ or newer
- USB Webcam / Pi Camera (Minimum 480p)
- Battery Pack (5V 3A) for portability
- Speaker or Headphones for audio feedback
- Tactile Push Button (connected to GPIO)

## Software Requirements
- Python 3.8+
- TensorFlow / Keras (for training)
- tflite-runtime (for deployment)
- OpenCV (opencv-python)
- pyttsx3 (for offline text-to-speech)
- Pillow (PIL)

## Dataset Structure
Organize the training data into the following directory structure:
```text
dataset/
├── 10/
├── 20/
├── 50/
├── 100/
├── 200/
├── 500/
└── others/
```

## Implementation Logic

- **Capture:** The system waits for a hardware interrupt (button press).  
- **Pre-process:** The image is captured at 480p, resized to 224x224, and normalized to the range [0, 1] or [-1, 1] as required by MobileNetV2.  
- **Inference:** The TFLite interpreter loads the model into the Pi's RAM and performs a forward pass.  
- **Classification:** The softmax output identifies the denomination with the highest probability.  
- **Feedback:** The system checks if the confidence score exceeds a pre-defined threshold (e.g., 0.8) before announcing the value via the speaker.  

## Team Members

- Achal Nanjundamurthy (1JT21CS003)  
- Aruna A Shenoy (1JT21CS019)  
- Bhushan M V (1JT21CS030)  
- Pujitha D R (1JT21CS128)  

Project guided by Dr. Prabhanjan S., HOD, Dept. of CSE, Jyothy Institute of Technology.
