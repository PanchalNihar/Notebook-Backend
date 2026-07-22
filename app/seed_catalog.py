import logging
from typing import List, Dict, Any
from app.models import CachedTrack

DEFAULT_SEED_TRACKS: List[Dict[str, Any]] = [
    # Happy / Party
    {
        "emotion": "Happy",
        "spotify_id": "1vC62R8pM3iW8G5e27H5rF",
        "name": "Gallan Goodiyaan",
        "artist": "Yashita Sharma, Manish Kumar Tipu, Farhan Akhtar",
        "album": "Dil Dhadakne Do",
        "album_art_url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/1vC62R8pM3iW8G5e27H5rF",
        "duration_ms": 296000,
        "explicit": False,
        "popularity": 75,
        "release_date": "2015-05-02"
    },
    {
        "emotion": "Happy",
        "spotify_id": "67406R9c5H9S93Z8u190",
        "name": "Nashe Si Chhod Gayi",
        "artist": "Arijit Singh, Caralisa Monteiro",
        "album": "Befikre",
        "album_art_url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/67406R9c5H9S93Z8u190",
        "duration_ms": 237000,
        "explicit": False,
        "popularity": 78,
        "release_date": "2016-11-03"
    },
    {
        "emotion": "Happy",
        "spotify_id": "0GneL3y1aW4517V97N",
        "name": "London Thumakda",
        "artist": "Labh Janjua, Sonu Kakkar, Neha Kakkar",
        "album": "Queen",
        "album_art_url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/0GneL3y1aW4517V97N",
        "duration_ms": 230000,
        "explicit": False,
        "popularity": 74,
        "release_date": "2014-01-25"
    },

    # Romantic
    {
        "emotion": "Romantic",
        "spotify_id": "8R001N93471Vb3719P",
        "name": "Tum Hi Ho",
        "artist": "Arijit Singh",
        "album": "Aashiqui 2",
        "album_art_url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/8R001N93471Vb3719P",
        "duration_ms": 262000,
        "explicit": False,
        "popularity": 88,
        "release_date": "2013-04-06"
    },
    {
        "emotion": "Romantic",
        "spotify_id": "8R002N93471Vb3719P",
        "name": "Kesariya",
        "artist": "Arijit Singh, Pritam",
        "album": "Brahmastra",
        "album_art_url": "https://images.unsplash.com/photo-1516450360452-9312f5e86fc7?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/8R002N93471Vb3719P",
        "duration_ms": 268000,
        "explicit": False,
        "popularity": 90,
        "release_date": "2022-07-17"
    },
    {
        "emotion": "Romantic",
        "spotify_id": "8R003N93471Vb3719P",
        "name": "Raataan Lambiyan",
        "artist": "Tanishk Bagchi, Jubin Nautiyal, Asees Kaur",
        "album": "Shershaah",
        "album_art_url": "https://images.unsplash.com/photo-1445307806294-bff7f67ff225?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/8R003N93471Vb3719P",
        "duration_ms": 230000,
        "explicit": False,
        "popularity": 87,
        "release_date": "2021-07-30"
    },

    # Gym / Energetic
    {
        "emotion": "Gym",
        "spotify_id": "9G001N93471Vb3719P",
        "name": "Ziddi Dil",
        "artist": "Vishal Dadlani",
        "album": "Mary Kom",
        "album_art_url": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/9G001N93471Vb3719P",
        "duration_ms": 286000,
        "explicit": False,
        "popularity": 81,
        "release_date": "2014-08-13"
    },
    {
        "emotion": "Gym",
        "spotify_id": "9G002N93471Vb3719P",
        "name": "Brothers Anthem",
        "artist": "Vishal Dadlani",
        "album": "Brothers",
        "album_art_url": "https://images.unsplash.com/photo-1465847899084-d164df4dedc6?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/9G002N93471Vb3719P",
        "duration_ms": 353000,
        "explicit": False,
        "popularity": 83,
        "release_date": "2015-07-28"
    },
    {
        "emotion": "Gym",
        "spotify_id": "9G003N93471Vb3719P",
        "name": "Sultan Title Track",
        "artist": "Sukhwinder Singh, Shadab Faridi",
        "album": "Sultan",
        "album_art_url": "https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/9G003N93471Vb3719P",
        "duration_ms": 280000,
        "explicit": False,
        "popularity": 85,
        "release_date": "2016-05-31"
    },

    # Sad
    {
        "emotion": "Sad",
        "spotify_id": "1c74128N70258VbN97",
        "name": "Channa Mereya",
        "artist": "Arijit Singh",
        "album": "Ae Dil Hai Mushkil",
        "album_art_url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/1c74128N70258VbN97",
        "duration_ms": 289000,
        "explicit": False,
        "popularity": 82,
        "release_date": "2016-09-29"
    },
    {
        "emotion": "Sad",
        "spotify_id": "4G716N93471Vb3719P",
        "name": "Agar Tum Saath Ho",
        "artist": "Alka Yagnik, Arijit Singh",
        "album": "Tamasha",
        "album_art_url": "https://images.unsplash.com/photo-1445307806294-bff7f67ff225?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/4G716N93471Vb3719P",
        "duration_ms": 341000,
        "explicit": False,
        "popularity": 85,
        "release_date": "2015-10-27"
    },

    # Neutral / Calm
    {
        "emotion": "Neutral",
        "spotify_id": "3G934716Vb37N77P11",
        "name": "Iktara",
        "artist": "Kavita Seth, Amitabh Bhattacharya",
        "album": "Wake Up Sid",
        "album_art_url": "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/3G934716Vb37N77P11",
        "duration_ms": 253000,
        "explicit": False,
        "popularity": 76,
        "release_date": "2009-08-21"
    },
    {
        "emotion": "Neutral",
        "spotify_id": "5G934716Vb37N77P33",
        "name": "Kho Gaye Hum Kahan",
        "artist": "Jasleen Royal, Prateek Kuhad",
        "album": "Baar Baar Dekho",
        "album_art_url": "https://images.unsplash.com/photo-1487180144351-b8472da7d491?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/5G934716Vb37N77P33",
        "duration_ms": 213000,
        "explicit": False,
        "popularity": 78,
        "release_date": "2016-08-03"
    },

    # Angry
    {
        "emotion": "Angry",
        "spotify_id": "6G934716Vb37N77P44",
        "name": "Sadda Haq",
        "artist": "Mohit Chauhan",
        "album": "Rockstar",
        "album_art_url": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/6G934716Vb37N77P44",
        "duration_ms": 365000,
        "explicit": False,
        "popularity": 80,
        "release_date": "2011-09-30"
    },
    {
        "emotion": "Angry",
        "spotify_id": "7G934716Vb37N77P55",
        "name": "Aarambh Hai Prachand",
        "artist": "Piyush Mishra",
        "album": "Gulaal",
        "album_art_url": "https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/7G934716Vb37N77P55",
        "duration_ms": 295000,
        "explicit": False,
        "popularity": 81,
        "release_date": "2009-02-27"
    },

    # Surprise
    {
        "emotion": "Surprise",
        "spotify_id": "9G934716Vb37N77P77",
        "name": "Matargashti",
        "artist": "Mohit Chauhan",
        "album": "Tamasha",
        "album_art_url": "https://images.unsplash.com/photo-1511735111819-9a3f7709049c?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/9G934716Vb37N77P77",
        "duration_ms": 328000,
        "explicit": False,
        "popularity": 79,
        "release_date": "2015-10-27"
    },

    # Fear
    {
        "emotion": "Fear",
        "spotify_id": "2H934716Vb37N77P99",
        "name": "Kun Faya Kun",
        "artist": "AR Rahman, Javed Ali, Mohit Chauhan",
        "album": "Rockstar",
        "album_art_url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/2H934716Vb37N77P99",
        "duration_ms": 473000,
        "explicit": False,
        "popularity": 86,
        "release_date": "2011-09-30"
    },

    # Disgust
    {
        "emotion": "Disgust",
        "spotify_id": "4H934716Vb37N77P11",
        "name": "Bhaag D.K. Bose",
        "artist": "Ram Sampath",
        "album": "Delhi Belly",
        "album_art_url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300&auto=format&fit=crop&q=80",
        "preview_url": None,
        "spotify_url": "https://open.spotify.com/track/4H934716Vb37N77P11",
        "duration_ms": 243000,
        "explicit": True,
        "popularity": 74,
        "release_date": "2011-05-07"
    },
]


async def seed_cached_tracks():
    """Seed or update cached_tracks collection with expanded mood categories."""
    try:
        await CachedTrack.delete_all()
        logging.info("Seeding cached_tracks collection with multi-category fallback tracks...")
        tracks_to_insert = [CachedTrack(**t) for t in DEFAULT_SEED_TRACKS]
        await CachedTrack.insert_many(tracks_to_insert)
        logging.info(f"Successfully seeded {len(tracks_to_insert)} fallback tracks across emotions.")
    except Exception as e:
        logging.error(f"Error during cached_tracks seeding: {e}")
