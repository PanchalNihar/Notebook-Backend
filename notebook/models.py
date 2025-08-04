# models.py
from beanie import Document, Indexed
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional, List
import uuid
from enum import Enum

class DetectionMethod(str, Enum):
    IMAGE = "image"
    TEXT = "text"

class User(Document):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: Indexed(EmailStr, unique=True)
    username: Optional[str] = None
    password_hash: Optional[str]=None
    google_id: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: datetime = datetime.utcnow()
    is_active: bool = True
    
    class Settings:
        name = "users"

class MoodEntry(Document):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    emotion: str
    detection_method: DetectionMethod
    confidence_score: Optional[float] = None
    created_at: datetime = datetime.utcnow()
    
    class Settings:
        name = "mood_entries"

class RecommendedTrack(Document):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    mood_entry_id: str
    spotify_id: str
    track_name: str
    artist_name: str
    album_art_url: Optional[str] = None
    preview_url: Optional[str] = None
    spotify_url: Optional[str] = None
    user_rating: Optional[int] = None
    
    class Settings:
        name = "recommended_tracks"

class UserPlaylist(Document):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    spotify_playlist_id: str
    playlist_name: str
    emotion: str
    created_at: datetime = datetime.utcnow()
    
    class Settings:
        name = "user_playlists"

# Pydantic models for API requests/responses
class LoginRequest(BaseModel):
    email:str
    password:str
    
class RegisterRequest(BaseModel):
    email:str
    password:str
    username:str
    full_name: Optional[str] =None
    
class GoogleAuthRequest(BaseModel):
    credential: str

class UserResponse(BaseModel):
    id: str
    email: str
    username: Optional[str]
    full_name: Optional[str]
    avatar_url: Optional[str]

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse
