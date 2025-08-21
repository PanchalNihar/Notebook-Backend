import os
import logging
import time
from datetime import timedelta, datetime
from contextlib import asynccontextmanager
from typing import List, Optional
from random import randint
import cv2
import numpy as np
import spotipy
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    File,
    UploadFile,
    Depends,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from tensorflow.keras.models import load_model

# Local imports
from database import init_db
from auth import (
    verify_token,
    create_access_token,
    verify_google_token,
    get_or_create_google_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_user,
    get_password_hash
)
from models import (
    User,
    MoodEntry,
    RecommendedTrack,
    DetectionMethod,
    GoogleAuthRequest,
    AuthResponse,
    UserResponse,
    RegisterRequest,
    LoginRequest,
)


#Environment & Global Initialisation
load_dotenv()
logging.basicConfig(level=logging.INFO)

# FastAPI instance with lifespan for DB init
@asynccontextmanager
async def lifespan(app_: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="MoodTune API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS (adapt origins for production as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200","http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Spotify Initialisation
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

sp: Optional[spotipy.Spotify] = None
if SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET:
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        logging.info("Successfully connected to Spotify API.")
    except Exception as e:
        logging.error("Could not connect to Spotify API: %s", e)

# Emotion model & OpenCV#
try:
    emotion_model = load_model("face_emotion_model.keras")
    logging.info("Emotion detection model loaded.")
except Exception as e:
    emotion_model = None
    logging.error("Error loading emotion model: %s", e)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy",
    "Sad", "Surprise", "Neutral"
]

MOOD_MAPPING = {
    "Happy": "Bollywood party happy dance",
    "Sad": "sad Bollywood songs hindi acoustic",
    "Angry": "angry Bollywood rock",
    "Neutral": "Bollywood lo-fi chill study instrumental",
    "Surprise": "upbeat Bollywood retro 90s",
    "Fear": "soothing hindi instrumental calm",
    "Disgust": "intense hindi rock grunge",
    "Calm": "calm bollywood instrumental peaceful flute",
    "Energetic": "energetic punjabi party hits bhangra",
    "Romantic": "romantic bollywood hindi love songs",
    "DHH": "desi hip hop hindi rap",
}


#Pydantic Helper Models
class TrackOut(BaseModel):
    id: str  # Spotify track ID
    name: str
    artist: str
    album: str
    album_art_url: Optional[str] = None
    preview_url: Optional[str] = None
    has_preview: bool
    spotify_url: Optional[str] = None
    duration_ms: Optional[int] = None
    explicit: bool = False
    popularity: Optional[int] = None
    release_date: Optional[str] = None


class RecResponse(BaseModel):
    emotion: str
    confidence: Optional[float] = None
    tracks: List[TrackOut]
    mood_entry_id: Optional[str] = None


#Utility Functions
def preprocess_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Detect a face, crop, normalise and reshape for model."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (48, 48))
    roi = roi / 255.0
    return np.reshape(roi, (1, 48, 48, 1))


def get_spotify_tracks(emotion: str, offset: int = 0) -> List[TrackOut]:
    """Enhanced Spotify search with more track details."""
    if sp is None:
        raise HTTPException(status_code=503, detail="Spotify API not available.")

    query = MOOD_MAPPING.get(emotion, f"bollywood {emotion}")
    try:
        results = sp.search(q=query, type="track", limit=20, offset=offset, market="IN")
    except Exception as e:
        logging.error("Spotify search error: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching from Spotify.")

    items = results["tracks"]["items"]
    if not items:
        raise HTTPException(status_code=404, detail=f"No tracks found for emotion: {emotion}")

    tracks: List[TrackOut] = []
    for t in items:
        tracks.append(TrackOut(
            id=t["id"],
            name=t["name"],
            artist=", ".join(a["name"] for a in t["artists"]),
            album=t["album"]["name"],
            album_art_url=t["album"]["images"][0]["url"] if t["album"]["images"] else None,
            preview_url=t.get("preview_url"),
            has_preview=bool(t.get("preview_url")),
            spotify_url=t.get("external_urls", {}).get("spotify"),
            duration_ms=t.get("duration_ms"),
            explicit=t.get("explicit", False),
            popularity=t.get("popularity"),
            release_date=t["album"].get("release_date")
        ))
    return tracks


#Auth / Security Dependencies
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    token = credentials.credentials
    email = verify_token(token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    user = await User.find_one(User.email == email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user



#Endpoints

# Google OAuth 
@app.post("/auth/google", response_model=AuthResponse)
async def google_auth(body: GoogleAuthRequest):
    try:
        g_user = await verify_google_token(body.credential)
        user = await get_or_create_google_user(g_user)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = create_access_token(
            data={"sub": user.email},
            expires_delta=access_token_expires,
        )
        return AuthResponse(
            access_token=token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=user.email,
                username=user.username,
                full_name=user.full_name,
                avatar_url=user.avatar_url,
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Google authentication failed: {e}",
        )

@app.post("/auth/register", response_model=AuthResponse)
async def register_user(register_data: RegisterRequest):
    try:
        user = await create_user(register_data)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = create_access_token(
            data={"sub": user.email},
            expires_delta=access_token_expires,
        )
        
        return AuthResponse(
            access_token=token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=user.email,
                username=user.username,
                full_name=user.full_name,
                avatar_url=user.avatar_url,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

@app.post("/auth/login",response_model=AuthResponse)
async def login_user(login_data:LoginRequest):
    user = await authenticate_user(login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires,
    )
    
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            avatar_url=user.avatar_url,
        ),
    )
# Recommendations ---------------------------------------------------- #
@app.post("/api/recommendations/by-image", response_model=RecResponse)
async def recommend_by_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    if emotion_model is None:
        raise HTTPException(
            status_code=503,
            detail="Emotion model not available."
        )

    img_bytes = await file.read()
    processed = preprocess_image(img_bytes)
    if processed is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in image."
        )

    preds = emotion_model.predict(processed)
    predicted_emotion = EMOTION_LABELS[int(np.argmax(preds))]
    confidence = float(np.max(preds))

    tracks = get_spotify_tracks(predicted_emotion)

    # Save mood entry & tracks to MongoDB ---------------------------
    mood_entry = MoodEntry(
        user_id=current_user.id,
        emotion=predicted_emotion,
        detection_method=DetectionMethod.IMAGE,
        confidence_score=confidence,
    )
    await mood_entry.insert()

    for t in tracks:
        await RecommendedTrack(
            mood_entry_id=mood_entry.id,
            spotify_id=t.spotify_url or "",
            track_name=t.name,
            artist_name=t.artist,
            album_art_url=t.album_art_url,
            preview_url=t.preview_url,
            spotify_url=t.spotify_url,
        ).insert()

    return RecResponse(
        emotion=predicted_emotion,
        confidence=confidence,
        tracks=tracks,
        mood_entry_id=mood_entry.id,
    )


@app.get(
    "/api/recommendations/by-text/{emotion_text}",
    response_model=RecResponse
)
async def recommend_by_text(
    emotion_text: str,
    offset = randint(0,800),
    current_user: User = Depends(get_current_user),
):
    normalized = emotion_text.strip().capitalize()
    tracks = get_spotify_tracks(normalized,offset=offset)

    mood_entry = MoodEntry(
        user_id=current_user.id,
        emotion=normalized,
        detection_method=DetectionMethod.TEXT,
    )
    await mood_entry.insert()

    for t in tracks:
        await RecommendedTrack(
            mood_entry_id=mood_entry.id,
            spotify_id=t.spotify_url or "",
            track_name=t.name,
            artist_name=t.artist,
            album_art_url=t.album_art_url,
            preview_url=t.preview_url,
            spotify_url=t.spotify_url,
        ).insert()

    return RecResponse(
        emotion=normalized,
        tracks=tracks,
        mood_entry_id=mood_entry.id,
    )


# Analytics ---------------------------------------------------------- #
@app.get("/api/analytics/mood-history")
async def mood_history(
    days: int = 30,
    current_user: User = Depends(get_current_user),
):
    """Return aggregated mood data for the past `days` days."""
    from datetime import datetime, timedelta

    start_date = datetime.utcnow() - timedelta(days=days)

    pipeline = [
        {
            "$match": {
                "user_id": current_user.id,
                "created_at": {"$gte": start_date},
            }
        },
        {
            "$group": {
                "_id": {
                    "date": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$created_at",
                        }
                    },
                    "emotion": "$emotion",
                },
                "count": {"$sum": 1},
            }
        },
        {
            "$group": {
                "_id": "$_id.emotion",
                "total_count": {"$sum": "$count"},
                "dates": {
                    "$push": {
                        "date": "$_id.date",
                        "count": "$count",
                    }
                },
            }
        },
    ]

    # Fix: Use the cursor correctly with Beanie
    try:
        # Method 1: Use async iteration
        result = []
        async for doc in MoodEntry.aggregate(pipeline):
            result.append(doc)
    except Exception as e:
        # Method 2: Alternative approach using find method
        print(f"Aggregation error: {e}")
        # Fallback: Get recent mood entries and process them manually
        recent_entries = await MoodEntry.find(
            MoodEntry.user_id == current_user.id,
            MoodEntry.created_at >= start_date
        ).to_list()
        
        # Process manually
        emotion_distribution: dict[str, int] = {}
        mood_timeline: dict[str, dict[str, int]] = {}
        total_entries = len(recent_entries)
        
        for entry in recent_entries:
            emotion = entry.emotion
            date_str = entry.created_at.strftime("%Y-%m-%d")
            
            # Update emotion distribution
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            
            # Update mood timeline
            if date_str not in mood_timeline:
                mood_timeline[date_str] = {}
            mood_timeline[date_str][emotion] = mood_timeline[date_str].get(emotion, 0) + 1
        
        return {
            "emotion_distribution": emotion_distribution,
            "total_entries": total_entries,
            "mood_timeline": mood_timeline,
            "date_range": {
                "start": start_date.isoformat(),
                "end": datetime.utcnow().isoformat(),
            },
        }

    # Process aggregation results
    emotion_distribution: dict[str, int] = {}
    mood_timeline: dict[str, dict[str, int]] = {}
    total_entries = 0

    for doc in result:
        emotion = doc["_id"]
        total = doc["total_count"]
        emotion_distribution[emotion] = total
        total_entries += total
        for d in doc["dates"]:
            date = d["date"]
            mood_timeline.setdefault(date, {})
            mood_timeline[date][emotion] = d["count"]

    return {
        "emotion_distribution": emotion_distribution,
        "total_entries": total_entries,
        "mood_timeline": mood_timeline,
        "date_range": {
            "start": start_date.isoformat(),
            "end": datetime.utcnow().isoformat(),
        },
    }