# database.py
import os
import motor.motor_asyncio
from beanie import init_beanie
from models import User, MoodEntry, RecommendedTrack, UserPlaylist

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/moodtune")

async def init_db():
    # Create Motor client
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    
    # Initialize beanie with the document models
    await init_beanie(
        database=client.moodtune,
        document_models=[User, MoodEntry, RecommendedTrack, UserPlaylist]
    )

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
