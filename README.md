AI Mood DJ: Real-Time Emotion-Based Music Player
Project Description
AI Mood DJ is an end-to-end computer vision application that synchronizes audio playback with human emotions. The system utilizes a custom Deep Convolutional Neural Network (CNN) to analyze facial expressions from a live webcam feed and maps the detected sentiment to curated Spotify playlists. The project demonstrates the integration of deep learning, real-time image processing, and third-party API interaction.

Core Components
1. Emotion Recognition Engine
The backbone of the project is a deep learning model trained on facial expression datasets.

Architecture: A VGG-style CNN featuring 5 convolutional blocks.

Feature Extraction: Progressively increases channels (from 32 to 512) to capture complex facial features.

Regularization: Employs Batch Normalization, Dropout (0.5), and Global Average Pooling to ensure high generalization and prevent overfitting.

Output: Classifies input into 7 categories: Happy, Sad, Angry, Fear, Surprise, Neutral, and Disgust.

2. Real-Time Pipeline
Face Detection: Uses OpenCV to isolate facial regions from the video stream.

Inference: The processed frame is passed to the PyTorch model for real-time classification.

Stabilization: Implements a smoothing mechanism (using deques/averaging) to prevent flickering between emotions.

3. Spotify Automation
API Integration: Connects via the Spotipy library using the OAuth 2.0 flow.

Dynamic Triggering: Maps the dominant emotion to a specific Playlist URI. When the user triggers the action, the system opens the web browser to the corresponding music.

Technical Specifications
Framework: PyTorch

Loss Function: Label Smoothing Cross-Entropy (to improve model robustness).

Optimizer: AdamW with Weight Decay.

Image Transforms: Resize (128x128), Random Horizontal Flip, Rotation, and Normalization.

Prerequisites and Installation
Ensure you have Python 3.8+ installed.

Install Dependencies:

Bash

pip install torch torchvision opencv-python spotipy seaborn scikit-learn tqdm
Spotify Credentials: Obtain a CLIENT_ID and CLIENT_SECRET from the Spotify Developer Dashboard and add them to the configuration block in the script.

System Workflow
Initialize the camera and load the pre-trained weights (custom_emotion_model.pth).

The UI displays the live feed with a bounding box around the face and the predicted emotion label.

Press 'P' to fetch the Spotify playlist associated with the current mood.

Press 'Q' to safely release the camera and close the application.

Performance Evaluation
The model's reliability is verified through:

Confusion Matrix: To visualize class-wise performance and inter-class confusion.

Classification Report: Providing Precision, Recall, and F1-Scores for each emotion.
