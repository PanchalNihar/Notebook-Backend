import os
import io
import logging
from dotenv import load_dotenv

import cv2
import numpy as np
import spotipy
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from spotipy.oauth2 import SpotifyClientCredentials
from tensorflow.keras.models import load_model

# --- Basic Setup ---
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)


# --- CORS Middleware ---
origins = ["http://localhost:4200"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Load Models and API Clients at Startup ---
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
sp = None
if client_id and client_secret:
    try:
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth_manager)
        logging.info("Successfully connected to Spotify API.")
    except Exception as e:
        logging.error(f"Could not connect to Spotify API: {e}")

try:
    emotion_model = load_model("face_emotion_model.keras")
    logging.info("Emotion detection model loaded successfully.")
except Exception as e:
    emotion_model = None
    logging.error(f"Error loading emotion detection model: {e}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# --- Constants and Mappings ---
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MOOD_MAPPING = {
    'Happy': 'Bollywood party happy dance',
    'Sad': 'sad Bollywood songs hindi acoustic',
    'Angry': 'angry Bollywood rock',
    'Neutral': 'Bollywood lo-fi chill study instrumental',
    'Surprise': 'upbeat Bollywood retro 90s',
    'Fear': 'soothing hindi instrumental calm',
    'Disgust': 'intense hindi rock grunge',
    'Calm': 'calm bollywood instrumental peaceful flute',
    'Energetic': 'energetic punjabi party hits bhangra',
    'Romantic': 'romantic bollywood hindi love songs',
    'DHH': 'desi hip hop hindi rap'
}


# --- Pydantic Models ---
class Track(BaseModel):
    name: str
    artist: str
    album_art_url: str | None = None
    preview_url: str | None = None
    has_preview: bool
    spotify_url: str | None = None

class MusicRecommendationResponse(BaseModel):
    emotion: str
    tracks: list[Track]


# --- Helper Functions ---
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) == 0: return None
    (x, y, w, h) = faces[0]
    face_roi = gray_img[y:y+h, x:x+w]
    resized_face = cv2.resize(face_roi, (48, 48))
    normalized_face = resized_face / 255.0
    return np.reshape(normalized_face, (1, 48, 48, 1))

def get_spotify_tracks(emotion_text: str):
    search_query = MOOD_MAPPING.get(emotion_text, f"bollywood {emotion_text}")
    try:
        results = sp.search(q=search_query, type='track', limit=20, market="IN")
        if not results['tracks']['items']:
            raise HTTPException(status_code=404, detail=f"Could not find tracks for mood: {emotion_text}")
        
        recommendations = []
        for track in results['tracks']['items']:
            recommendations.append(Track(
                name=track['name'],
                artist=', '.join([artist['name'] for artist in track['artists']]),
                album_art_url=track['album']['images'][0]['url'] if track['album']['images'] else None,
                preview_url=track.get('preview_url'),
                has_preview=True if track.get('preview_url') else False,
                spotify_url=track.get('external_urls', {}).get('spotify')
            ))
        return recommendations
    except Exception as e:
        logging.error(f"Spotify search error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching from Spotify.")


# --- API Endpoints ---
@app.post("/by-image", response_model=MusicRecommendationResponse)
async def recommend_by_image(file: UploadFile = File(...)):
    if emotion_model is None: raise HTTPException(status_code=503, detail="AI model not available.")
    
    image_bytes = await file.read()
    processed_face = preprocess_image(image_bytes)
    if processed_face is None: raise HTTPException(status_code=400, detail="Could not detect a face.")

    predictions = emotion_model.predict(processed_face)
    predicted_emotion = EMOTION_LABELS[np.argmax(predictions)]
    
    tracks = get_spotify_tracks(predicted_emotion)
    return MusicRecommendationResponse(emotion=predicted_emotion, tracks=tracks)


# NEW ENDPOINT FOR TEXT SEARCH
@app.get("/by-text/{emotion_text}", response_model=MusicRecommendationResponse)
async def recommend_by_text(emotion_text: str):
    if sp is None: raise HTTPException(status_code=503, detail="Spotify API not available.")
    
    # Capitalize the first letter for better display and matching
    normalized_emotion = emotion_text.strip().capitalize()
    
    tracks = get_spotify_tracks(normalized_emotion)
    return MusicRecommendationResponse(emotion=normalized_emotion, tracks=tracks)