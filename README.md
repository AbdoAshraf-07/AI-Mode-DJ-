AI Mood DJ: Real-Time Emotion-Based Music Player
Project Overview
This project is an intelligent music recommendation system that combines Computer Vision and Deep Learning to detect facial expressions in real-time. Based on the detected emotion, the system interacts with the Spotify API to automatically select and play music playlists that match the user's current mood.

Key Features
Emotion Detection: Utilizes a Deep Convolutional Neural Network (CNN) built with PyTorch.

Real-Time Analysis: Uses OpenCV for live webcam feed processing and face detection.

Spotify Integration: Connects to the Spotify Web API via the Spotipy library for automated playback.

Comprehensive Evaluation: Includes performance metrics such as Confusion Matrices and Classification Reports.

Technical Architecture
The system is built on a VGG-style CNN architecture with the following specifications:

Feature Extraction: 5 Convolutional blocks with Batch Normalization and ReLU activation.

Regularization: Global Average Pooling and Dropout layers to prevent overfitting.

Optimization: AdamW optimizer with Label Smoothing Cross-Entropy loss.

Installation
Clone the repository to your local machine.

Install the required dependencies:

Bash

pip install torch torchvision opencv-python spotipy seaborn scikit-learn tqdm
Set up your Spotify Developer credentials (Client ID and Client Secret) in the configuration section of the code.

Usage
Run the main application script or the Jupyter Notebook.

Ensure your webcam is connected and functional.

Press the 'P' key to capture your current emotion and trigger a corresponding Spotify playlist.

Press the 'Q' key to exit the application.

Model Performance
The model classifies seven distinct emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. Detailed accuracy and loss curves, as well as per-class precision and recall, are provided in the evaluation section of the notebook.
