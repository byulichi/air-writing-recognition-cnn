# air-writing-recognition-cnn
A real-time computer vision system to assist dyslexic learners with character recognition
# Air Writing Recognition System for Dyslexic-Friendly Learning

## 📌 Project Overview
This project is a real-time computer vision application designed to assist dyslexic learners. It allows users to write characters in the air using their index finger. The system tracks the hand gestures using MediaPipe, renders the digital ink on a virtual OpenCV canvas, and classifies the written letters using a custom-trained Convolutional Neural Network (CNN). 

This system specifically focuses on mitigating common challenges in dyslexic learning, such as distinguishing between confusing character pairs (e.g., 'b' vs. 'd', and 'p' vs. 'q').

## 🛠️ Technology Stack
* **Language:** Python 3.11
* **Computer Vision:** OpenCV, MediaPipe (v0.10.5)
* **Deep Learning:** TensorFlow / Keras (v2.15.0)
* **Dataset:** EMNIST (Extended MNIST - Letters)
* **Data Visualization:** Tableau (Used for testing and system evaluation)

## ✨ Key Features
* **Real-Time Spatial Tracking:** Robust hand-tracking across various lighting conditions.
* **Smart Gesture Control:** * ☝️ **DRAW Mode:** Raise index finger only to write on the canvas.
  * ✌️ **PAUSE Mode:** Raise index and middle fingers to move your hand without drawing.
* **High-Accuracy Classification:** A lightweight CNN architecture optimized to prevent overfitting via Early Stopping, achieving high Precision and Recall on target letter pairs.

## 🚀 How to Run the Project

### 1. Environment Setup
It is highly recommended to run this system in an isolated virtual environment to prevent dependency conflicts.
```bash
# Create a Python 3.11 virtual environment

### 2. Install Dependencies
Install the strictly required library versions:
pip install tensorflow==2.15.0 mediapipe==0.10.5 opencv-python pandas numpy

### 3. Usage
First, run the training script to generate the CNN model (requires EMNIST dataset CSVs):

python train_cnn.py
Once the dyslexia_air_writing_model.h5 file is generated, launch the interactive canvas:
python air_canvas.py
A webcam window will appear. Wave your hand to begin drawing. Press c to clear the canvas and q to quit.

📊 System Evaluation
(Note: Tableau dashboards detailing the confusion matrix and tracking robustness tests will be added here upon final system validation).

