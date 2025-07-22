# main.py
import os
import io
import logging
from dotenv import load_dotenv

import cv2 
import numpy as np
import spotipy
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from spotipy.oauth2 import SpotifyClientCredentials
from tensorflow.keras.models import load_model

# --- Basic Setup ---
# Load environment variables from the .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)


# --- Load Models and API Clients at Startup ---
# This is done once when the server starts for better performance

# 1. Load Spotify API Client
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
else:
    logging.error("Spotify credentials not found. Make sure you have a .env file.")

# 2. Load the trained Emotion Detection Model
try:
    emotion_model = load_model("face_emotion_model.keras")
    logging.info("Emotion detection model loaded successfully.")
except Exception as e:
    emotion_model = None
    logging.error(f"Error loading emotion detection model: {e}")

# 3. Load the face detection classifier from OpenCV
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
    
    # --- NEW MOODS ---
    'Calm': 'calm bollywood instrumental peaceful flute',
    'Energetic': 'energetic punjabi party hits bhangra',
    'Romantic': 'romantic bollywood hindi love songs',

    # --- Extended Moods ---
    'Excited': 'bollywood high energy workout hits',
    'Tired': 'relaxing hindi unplugged soft songs',
    'Joyful': 'joyful bollywood wedding sangeet songs',
    'Lonely': 'hindi sad solo acoustic emotional',
    'Nostalgic': 'old hindi classics retro golden era',
    'Motivated': 'bollywood motivational inspirational tracks',
    'Heartbroken': 'bollywood heartbreak breakup hindi sad songs',
    'Hopeful': 'uplifting bollywood hindi soft pop',
    'Confused': 'ambient bollywood chill experimental',
    'Shy': 'gentle romantic bollywood slow love songs',
    'Bored': 'fun bollywood peppy dance numbers',
    'Focused': 'hindi lo-fi focus chill study beats',
    'Grateful': 'positive bollywood devotional peaceful songs',
    'Determined': 'intense bollywood workout gym beats',
    'Peaceful': 'bollywood flute instrumental sitar calm meditative',
    'Desi Hip Hop': 'desi hip hop hindi rap punjabi rap'
}


# --- Pydantic Models for API Response ---
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


# --- Image Preprocessing Function ---
def preprocess_image(image_bytes):
    """
    Reads image bytes, detects a face, processes it, and prepares it for the model.
    """
    # Convert bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) == 0:
        return None  # No face found

    # Assume the first face is the one we want and crop it
    (x, y, w, h) = faces[0]
    face_roi = gray_img[y:y+h, x:x+w]

    # Resize, normalize, and reshape for the model
    resized_face = cv2.resize(face_roi, (48, 48))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

    return reshaped_face


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Mood Music API is running!"}

@app.post("/get-music-recommendation", response_model=MusicRecommendationResponse)
async def get_music_recommendation(file: UploadFile = File(...)):
    """
    The main endpoint. Receives an image, predicts emotion, and returns music.
    """
    if emotion_model is None:
        raise HTTPException(status_code=503, detail="Emotion detection model is not available.")
    if sp is None:
        raise HTTPException(status_code=503, detail="Spotify API is not available.")

    # 1. Read and preprocess the image
    image_bytes = await file.read()
    processed_face = preprocess_image(image_bytes)

    if processed_face is None:
        raise HTTPException(status_code=400, detail="Could not detect a face in the uploaded image.")

    # 2. Predict the emotion
    predictions = emotion_model.predict(processed_face)
    predicted_emotion_index = np.argmax(predictions)
    predicted_emotion = EMOTION_LABELS[predicted_emotion_index]

    # 3. Get music recommendations from Spotify
    search_query = MOOD_MAPPING.get(predicted_emotion, "bollywood chill") # Default search
    try:
        results = sp.search(q=search_query, type='track', limit=20, market="IN")
        if not results['tracks']['items']:
            raise HTTPException(status_code=404, detail=f"Could not find tracks for emotion: {predicted_emotion}")

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
        
        return MusicRecommendationResponse(emotion=predicted_emotion, tracks=recommendations)

    except Exception as e:
        logging.error(f"An error occurred during Spotify search: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching music recommendations.")

