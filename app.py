from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
import random
import threading

app = Flask(__name__)

# --- 1. CONFIGURATION ---
CLIENT_ID = '3abff4a0aff5405781375df50b8b4a98'
CLIENT_SECRET = '075bb924d86d4bce9145ec4edf121b72'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Surprise', 'Disgust', 'Angry', 'Happy', 'Neutral', 'Sad', 'Neutral']

# Global State
current_mood_state = {"mood": "Neutral", "confidence": 0.0}

# --- 2. CLASSES DEFINITION ---
class CustomDeepEmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomDeepEmotionNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SpotifyPlaylistMaker:
    def __init__(self, client_id, client_secret):
        scope = "playlist-modify-public user-library-read"
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id, client_secret=client_secret,
            redirect_uri="http://127.0.0.1:8888/callback", scope=scope
        ))
        try:
            self.user_id = self.sp.current_user()["id"]
            print(f"Spotify Connected: {self.user_id}")
        except Exception as e: # Catching specific exception
            print(f"Spotify Connection Failed: {e}")
        
        self.mood_queries = {
            "Happy": [
                "أغاني شعبي مصري حكيم",       # Hakim/Classic Shaabi
                "مهرجانات رقص",              # Mahraganat (Festivals)
                "عمرو دياب وتامر حسني",      # Egyptian Pop Kings
                "أغاني أفراح مصرية"          # Egyptian Wedding Songs
            ],
            "Sad": [
                "تامر عاشور نكد",            # Tamer Ashour (The King of Sadness)
                "شيرين عبد الوهاب دراما",     # Sherine Drama
                "جورج وسوف سلطنة",           # George Wassouf (Very popular for sad moods)
                "أغاني فراق وحزن مصري"       # General breakup songs
            ],
            "Angry": [
                "ويجز ومروان بابلو",         # Egyptian Trap/Rap (Aggressive)
                "كايروكي روك",               # Cairokee (Rock/Revolutionary)
                "مهرجانات ضرب نار",          # Heavy Mahraganat
                "أغاني راب سين"              # The Rap Scene
            ],
            "Neutral": [
                "محمد منير رايق",            # Mohamed Mounir (The King - Chill vibes)
                "فيروز صباحيات",             # Fairouz (Essential for Egyptian mornings)
                "مسار إجباري",               # Massar Egbari (Indie/Chill)
                "كلثوميات موسيقى"            # Umm Kulthum Instrumentals
            ],
            "Surprise": [
                "أغاني تسعينات مصري",        # 90s Nostalgia (Ehab Tawfik, Hisham Abbas)
                "ريمكسات شعبي",              # Shaabi Remixes
                "تريند تيك توك مصري",        # Viral Egyptian Trends
                "إعلانات رمضان القديمة"      # Old Ramadan Ads (Nostalgic surprise)
            ],
            "Fear": [
                "موسيقى الفيل الأزرق",       # Blue Elephant Soundtrack (Suspense)
                "موسيقى رأفت الهجان",        # Raafat Al Haggan (Classic Suspense)
                "موسيقى رعب وغموض"           # General Mystery
            ],
            "Disgust": [
                "أندر جراوند مصري",          # Egyptian Underground (Gritty)
                "ميتال مصري",                # Egyptian Metal (Niche but exists)
                "انتقاد مجتمعي راب"          # Social conscious/Critical Rap
            ]
        }

    def create_playlist(self, mood):
        if mood not in self.mood_queries: return None
        query = random.choice(self.mood_queries[mood])
        try:
            playlist = self.sp.user_playlist_create(
                user=self.user_id, name=f"AI Mood: {mood} 🎵", public=True,
                description=f"Generated by AI Mood DJ based on {mood} emotion."
            )
            results = self.sp.search(q=query, limit=20, type='track')
            uris = [t['uri'] for t in results['tracks']['items']]
            if uris:
                random.shuffle(uris)
                self.sp.playlist_add_items(playlist['id'], uris)
                return playlist['external_urls']['spotify']
        except Exception as e:
            print(f"Error: {e}")
            return None
        return None

# --- 3. INITIALIZATION ---
model = CustomDeepEmotionNet(num_classes=7).to(device)
try:
    model.load_state_dict(torch.load('custom_emotion_model.pth', map_location=device))
    model.eval()
    print("Model Loaded")
except Exception as e: # Catching specific exception
    print(f"Model Failed to load: {e}")

dj = SpotifyPlaylistMaker(CLIENT_ID, CLIENT_SECRET)
transform = transforms.Compose([
    transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 4. VIDEO STREAMING LOGIC ---
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        return

    print("Camera opened successfully.")
    while True:
        success, frame = cap.read()
        if not success: 
            print("Error: Failed to grab frame.")
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        detected_mood = "Scanning..."

        if len(faces) == 0:
            print("No faces detected.")
        
        for (x, y, w, h) in faces:
            print(f"Face detected at: x={x}, y={y}, w={w}, h={h}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            try:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0: continue
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(roi_rgb)
                tensor = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = model(tensor)
                    probs = F.softmax(out, dim=1).cpu().numpy().squeeze()
                
                idx = np.argmax(probs)
                detected_mood = class_names[idx]
                confidence = probs[idx] * 100
                
                # Update Global State
                current_mood_state["mood"] = detected_mood
                current_mood_state["confidence"] = float(confidence)
                
                # Draw on frame
                label_text = f"{detected_mood} ({confidence:.1f}%)"
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                print(f"Mood updated: {detected_mood}, Confidence: {confidence:.1f}%")
            except Exception as e:
                print(f"Error during mood detection: {e}")
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- 5. FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/create_playlist_api', methods=['POST'])
def create_playlist_api():
    mood = current_mood_state["mood"]
    if mood == "Scanning..." or mood == "Neutral":
        pass
    
    url = dj.create_playlist(mood)
    if url:
        return jsonify({"status": "success", "url": url, "mood": mood})
    else:
        return jsonify({"status": "error", "message": "Could not create playlist"})

@app.route('/get_current_mood')
def get_current_mood():
    return jsonify(current_mood_state)

if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(debug=True, port=5000)
