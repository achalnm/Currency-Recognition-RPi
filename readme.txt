cat << 'EOF' > README.md
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
