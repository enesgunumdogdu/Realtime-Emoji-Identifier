# 🎭 Real-time Emotion Identifier

A web application that analyzes facial expressions in real-time and identifies emotions using deep learning.

## ✨ Features

- 📹 Real-time camera feed with face detection
- 😊 Deep learning-based emotion recognition
- 🎯 Seven emotion categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- 🌐 Web-based interface with live video streaming
- 🤖 Pre-trained CNN model for accurate emotion detection

## 🚀 Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download the pre-trained model or train your own:
```bash
python train_model.py
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## 💡 Usage

1. 📸 Enable your web camera
2. 👤 Position your face in front of the camera
3. 🎭 View the real-time emotion detection results displayed on the video feed

## 📋 Requirements

- 🐍 Python 3.7+
- 📹 Web camera
- 🌐 Modern web browser
- 📚 Required Python packages:
  - TensorFlow
  - OpenCV
  - Flask
  - NumPy
  - Pandas
  - scikit-learn

## 🧠 Technical Details

- Uses a Convolutional Neural Network (CNN) for emotion recognition
- Trained on the FER2013 dataset
- Real-time face detection using OpenCV's Haar Cascade
- Web interface built with Flask
- Live video streaming with OpenCV
