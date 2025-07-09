import os
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

# Load environment variables from .env file
load_dotenv()

# Read MongoDB URI from environment variables
MONGO_URI = os.getenv("MONGO_URI")

# Create a MongoDB client with TLS (for MongoDB Atlas)
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())

def get_collection(db_name: str, collection_name: str):
    """
    Returns the MongoDB collection object from the specified database.
    """
    db = client[db_name]
    return db[collection_name]
