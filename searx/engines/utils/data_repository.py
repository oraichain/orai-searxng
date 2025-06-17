import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .mongodb import MongoDB

logger = logging.getLogger(__name__)
class DataRepository:
    def __init__(self, mongodb: MongoDB,collection_name: str):
        """
        Initialize the semantic question repository

        Args:
            mongodb: MongoDB connection instance
            embedding_service: Embedding service for generating vectors
        """
        self.mongodb = mongodb
        self.collection_name = collection_name

    def get_collection(self):
        """Get the MongoDB collection"""
        try:
            collection = self.mongodb.get_collection(self.collection_name)
            return collection
        except Exception as e:
            logger.error(f"Error getting collection '{self.collection_name}': {str(e)}")
            raise

    def query_by_category_id(self, category_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            # Kiểm tra đầu vào
            if not category_id or not category_id.strip():
                raise ValueError("category_id không được để trống")

            query = {"category_id": category_id.strip()}

            if start_time or end_time:
                time_query = {}
                if start_time:
                    time_query["$gte"] = start_time
                if end_time:
                    time_query["$lte"] = end_time
                if time_query:
                    query["created_at"] = time_query

            collection = self.get_collection()
            cursor = collection.find(query).limit(limit).sort("created_at", -1)

            results = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                results.append(doc)

            return results

        except Exception as e:
            logger.error(f"Lỗi khi truy vấn dữ liệu theo category_id: {str(e)}")
            raise
