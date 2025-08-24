import os
import motor.motor_asyncio
from beanie import init_beanie
from notebook.models import User, MoodEntry, RecommendedTrack, UserPlaylist
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL")
print("Mongo URL:", MONGODB_URL)

async def init_db():
    if not MONGODB_URL:
        raise ValueError("MONGODB_URL is not set in .env")

    # Create Motor client
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)

    # Initialize beanie with the document models
    await init_beanie(
        database=client.tuneify,  # this ensures DB name is 'tuneify'
        document_models=[User, MoodEntry, RecommendedTrack, UserPlaylist]
    )

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
