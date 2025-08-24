#auth.py
import os
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from google.auth.transport import requests
from google.oauth2 import id_token
from notebook.models import User, LoginRequest, RegisterRequest, AuthResponse, UserResponse
import uuid


SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        return email
    except JWTError:
        return None

# Traditional authentication
async def authenticate_user(email: str, password: str) -> User:
    """Authenticate user with email/password"""
    user = await User.find_one(User.email == email)
    if not user or not user.password_hash:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user

async def create_user(register_data: RegisterRequest) -> User:
    """Create new user with email/password"""
    # Check if user already exists
    existing_user = await User.find_one(User.email == register_data.email)
    if existing_user:
        raise ValueError("User with this email already exists")
    
    # Create new user
    user = User(
        id=str(uuid.uuid4()),
        email=register_data.email,
        username=register_data.username,
        password_hash=get_password_hash(register_data.password),
        full_name=register_data.full_name
    )
    await user.insert()
    return user

# Google OAuth (existing code)
async def verify_google_token(credential: str) -> dict:
    """Verify Google OAuth token and return user info"""
    try:
        from database import GOOGLE_CLIENT_ID
        idinfo = id_token.verify_oauth2_token(
            credential, requests.Request(), GOOGLE_CLIENT_ID
        )
        
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')
        
        return {
            'google_id': idinfo['sub'],
            'email': idinfo['email'],
            'full_name': idinfo.get('name'),
            'avatar_url': idinfo.get('picture'),
            'username': idinfo.get('given_name', idinfo['email'].split('@')[0])
        }
    except ValueError as e:
        raise Exception(f"Invalid Google token: {e}")

async def get_or_create_google_user(google_user_info: dict) -> User:
    """Get existing user or create new one from Google OAuth"""
    user = await User.find_one(User.google_id == google_user_info['google_id'])
    
    if not user:
        user = await User.find_one(User.email == google_user_info['email'])
        if user:
            # Update existing user with Google info
            user.google_id = google_user_info['google_id']
            user.full_name = google_user_info['full_name']
            user.avatar_url = google_user_info['avatar_url']
            if not user.username:
                user.username = google_user_info['username']
            await user.save()
        else:
            # Create new user
            user = User(
                id=str(uuid.uuid4()),
                email=google_user_info['email'],
                google_id=google_user_info['google_id'],
                username=google_user_info['username'],
                full_name=google_user_info['full_name'],
                avatar_url=google_user_info['avatar_url']
            )
            await user.insert()
    
    return user
