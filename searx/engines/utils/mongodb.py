from pymongo import MongoClient
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MongoDB:
    def __init__(self, uri: str):
        self.uri = uri
        self.client = None
        self.db = None

    def connect(self):
        """Connect to MongoDB and set up the database"""
        try:
            print(self.uri)
            self.client = MongoClient(self.uri)
            self.db = self.client['thesis_data_platform']
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def close(self):
        """Close the MongoDB connection"""
        if self.client is not None:
            self.client.close()
            logger.info("MongoDB connection closed")

    def get_collection(self, collection_name: Optional[str] = None):
        """Get the MongoDB collection"""
        if self.client is None:
            raise RuntimeError("MongoDB client not connected. Call connect() first.")

        if self.db is None:
            self.db = self.client['thesis_data_platform']

        return self.db[collection_name]

    def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        try:
            if self.client is not None:
                # Ping the server to check connection
                self.client.admin.command('ping')
                return True
        except Exception:
            pass
        return False
