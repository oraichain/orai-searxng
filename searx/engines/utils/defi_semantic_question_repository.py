from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from bson import ObjectId
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SemanticQuestionData(BaseModel):
    """Pydantic model for semantic question data"""
    question_text: str
    category_id: str
    category_name: str
    embedding_vector: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class DefiSemanticQuestionRepository:
    """Specialized repository for DeFi semantic questions with embedding support"""

    def __init__(self, mongodb, embedding_service=None, collection_vector: Optional[str] = None):
        """
        Initialize the semantic question repository

        Args:
            mongodb: MongoDB connection instance
            embedding_service: Embedding service for generating vectors
        """
        self.mongodb = mongodb
        self.embedding_service = embedding_service
        self.collection_name = collection_vector

    def get_collection(self):
        """Get the MongoDB collection"""
        try:
            collection = self.mongodb.get_collection(self.collection_name)
            return collection
        except Exception as e:
            logger.error(f"Error getting collection '{self.collection_name}': {str(e)}")
            raise

    # CREATE operations
    def add_question(self, question_text: str, category_id: str, category_name: str, auto_embed: bool = True) -> str:
        """
        Add a new semantic question to the collection

        Args:
            question_text: The question text
            category_id: Category identifier
            category_name: Category name
            auto_embed: Whether to automatically generate embedding

        Returns:
            Document ID of the inserted question
        """
        try:
            # Validate inputs
            if not question_text or not question_text.strip():
                raise ValueError("question_text cannot be empty")
            if not category_id or not category_id.strip():
                raise ValueError("category_id cannot be empty")
            if not category_name or not category_name.strip():
                raise ValueError("category_name cannot be empty")

            # Create the document structure
            question_doc = {
                "question_text": question_text.strip(),
                "category_id": category_id.strip(),
                "category_name": category_name.strip(),
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }

            # Generate embedding if requested and service is available
            if auto_embed and self.embedding_service is not None:
                try:
                    embedding_vector = self.embedding_service.embed_text(question_text)
                    question_doc["embedding_vector"] = embedding_vector
                    logger.info(f"Generated embedding for question: '{question_text[:50]}...'")
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {str(e)}")
                    question_doc["embedding_vector"] = None
            else:
                question_doc["embedding_vector"] = None

            # Insert into database
            collection = self.get_collection()
            result = collection.insert_one(question_doc)

            logger.info(f"Added semantic question with ID: {result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error adding semantic question: {str(e)}")
            raise

    def add_questions_batch(self, questions: List[Dict[str, str]], auto_embed: bool = True) -> List[str]:
        """
        Add multiple semantic questions in batch

        Args:
            questions: List of question dictionaries with keys: question_text, category_id, category_name
            auto_embed: Whether to automatically generate embeddings

        Returns:
            List of document IDs
        """
        try:
            if not questions:
                raise ValueError("questions list cannot be empty")

            # Validate all questions first
            for i, question in enumerate(questions):
                if not isinstance(question, dict):
                    raise ValueError(f"Question at index {i} must be a dictionary")

                required_keys = ["question_text", "category_id", "category_name"]
                for key in required_keys:
                    if key not in question or not question[key] or not str(question[key]).strip():
                        raise ValueError(f"Question at index {i} missing or empty '{key}'")

            question_docs = []

            # Prepare all questions for batch embedding if needed
            if auto_embed and self.embedding_service is not None:
                question_texts = [q["question_text"].strip() for q in questions]
                try:
                    embeddings = self.embedding_service.embed_texts(question_texts)
                    logger.info(f"Generated embeddings for {len(question_texts)} questions")
                except Exception as e:
                    logger.warning(f"Failed to generate batch embeddings: {str(e)}")
                    embeddings = [None] * len(questions)
            else:
                embeddings = [None] * len(questions)

            # Create documents
            for i, question in enumerate(questions):
                question_doc = {
                    "question_text": question["question_text"].strip(),
                    "category_id": question["category_id"].strip(),
                    "category_name": question["category_name"].strip(),
                    "embedding_vector": embeddings[i],
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                question_docs.append(question_doc)

            # Batch insert
            collection = self.get_collection()
            result = collection.insert_many(question_docs)
            document_ids = [str(id) for id in result.inserted_ids]

            logger.info(f"Added {len(document_ids)} semantic questions in batch")
            return document_ids

        except Exception as e:
            logger.error(f"Error adding semantic questions batch: {str(e)}")
            raise

    # READ operations
    def get_question_by_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Get a question by its ID"""
        try:
            if not question_id or not question_id.strip():
                raise ValueError("question_id cannot be empty")

            collection = self.get_collection()
            result = collection.find_one({"_id": ObjectId(question_id)})

            if result is not None:
                result['_id'] = str(result['_id'])

            return result

        except Exception as e:
            logger.error(f"Error getting question by ID: {str(e)}")
            raise

    def get_questions_by_category(self, category_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all questions in a specific category"""
        try:
            if not category_id or not category_id.strip():
                raise ValueError("category_id cannot be empty")

            collection = self.get_collection()
            cursor = collection.find({"category_id": category_id.strip()}).limit(limit)

            results = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                results.append(doc)

            return results

        except Exception as e:
            logger.error(f"Error getting questions by category: {str(e)}")
            raise

    def get_all_questions(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Get all questions with pagination"""
        try:
            collection = self.get_collection()
            cursor = collection.find().skip(skip).limit(limit).sort("created_at", -1)

            results = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                results.append(doc)

            return results

        except Exception as e:
            logger.error(f"Error getting all questions: {str(e)}")
            raise

    def search_questions_by_text(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search questions by text content"""
        try:
            if not search_text or not search_text.strip():
                raise ValueError("search_text cannot be empty")

            collection = self.get_collection()

            # Use regex search for text matching
            regex_pattern = {"$regex": search_text.strip(), "$options": "i"}  # Case insensitive
            cursor = collection.find({
                "$or": [
                    {"question_text": regex_pattern},
                    {"category_name": regex_pattern}
                ]
            }).limit(limit)

            results = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                results.append(doc)

            return results

        except Exception as e:
            logger.error(f"Error searching questions by text: {str(e)}")
            raise

    # SEMANTIC SEARCH operations
    def find_similar_questions(self, query_text: str, limit: int = 10, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find semantically similar questions using vector search

        Args:
            query_text: The query text to find similar questions for
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of similar questions with similarity scores
        """
        try:
            if not query_text or not query_text.strip():
                raise ValueError("query_text cannot be empty")

            if self.embedding_service is None:
                logger.warning("No embedding service available, falling back to text search")
                return self.search_questions_by_text(query_text, limit)

            # Generate embedding for the query
            query_embedding = self.embedding_service.embed_text(query_text.strip())
            logger.warning(f"using cosine similarity search")
            return self._cosine_similarity_search(query_embedding, limit, similarity_threshold)

        except Exception as e:
            logger.error(f"Error finding similar questions: {str(e)}")
            raise

    def _cosine_similarity_search(self, query_embedding: List[float], limit: int, threshold: float) -> List[Dict[str, Any]]:
        """Fallback cosine similarity search when vector search is not available"""
        try:
            import numpy as np
            from numpy.linalg import norm

            collection = self.get_collection()

            # Get all documents with embeddings
            cursor = collection.find({"embedding_vector": {"$exists": True, "$ne": None}})

            results_with_scores = []
            for doc in cursor:
                if doc.get("embedding_vector") is not None:
                    # Calculate cosine similarity
                    doc_embedding = np.array(doc["embedding_vector"])
                    query_vec = np.array(query_embedding)

                    similarity = np.dot(query_vec, doc_embedding) / (norm(query_vec) * norm(doc_embedding))

                    if similarity >= threshold:
                        doc['_id'] = str(doc['_id'])
                        doc['similarity_score'] = float(similarity)
                        results_with_scores.append(doc)

            # Sort by similarity score (descending) and limit results
            results_with_scores.sort(key=lambda x: x['similarity_score'], reverse=True)

            logger.info(f"Found {len(results_with_scores[:limit])} similar questions using cosine similarity")
            return results_with_scores[:limit]

        except ImportError:
            logger.error("NumPy not available for cosine similarity calculation")
            return []
        except Exception as e:
            logger.error(f"Error in cosine similarity search: {str(e)}")
            return []

    # UPDATE operations
    def update_question(self, question_id: str, update_data: Dict[str, Any], regenerate_embedding: bool = False) -> bool:
        """Update a question"""
        try:
            if not question_id or not question_id.strip():
                raise ValueError("question_id cannot be empty")

            if not update_data:
                raise ValueError("update_data cannot be empty")

            # Add updated timestamp
            update_data['updated_at'] = datetime.now()

            # Regenerate embedding if question text changed and requested
            if regenerate_embedding and "question_text" in update_data and self.embedding_service is not None:
                try:
                    new_embedding = self.embedding_service.embed_text(update_data["question_text"])
                    update_data["embedding_vector"] = new_embedding
                except Exception as e:
                    logger.warning(f"Failed to regenerate embedding: {str(e)}")

            collection = self.get_collection()
            result = collection.update_one(
                {"_id": ObjectId(question_id)},
                {"$set": update_data}
            )

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"Error updating question: {str(e)}")
            raise

    # DELETE operations
    def delete_question(self, question_id: str) -> bool:
        """Delete a question by ID"""
        try:
            if not question_id or not question_id.strip():
                raise ValueError("question_id cannot be empty")

            collection = self.get_collection()
            result = collection.delete_one({"_id": ObjectId(question_id)})

            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"Error deleting question: {str(e)}")
            raise

    def delete_questions_by_category(self, category_id: str) -> int:
        """Delete all questions in a category"""
        try:
            if not category_id or not category_id.strip():
                raise ValueError("category_id cannot be empty")

            collection = self.get_collection()
            result = collection.delete_many({"category_id": category_id.strip()})

            logger.info(f"Deleted {result.deleted_count} questions from category {category_id}")
            return result.deleted_count

        except Exception as e:
            logger.error(f"Error deleting questions by category: {str(e)}")
            raise

    # STATISTICS and UTILITY operations
    def get_question_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic questions collection"""
        try:
            collection = self.get_collection()

            # Basic counts
            total_questions = collection.count_documents({})
            questions_with_embeddings = collection.count_documents({"embedding_vector": {"$exists": True, "$ne": None}})

            # Category statistics
            category_pipeline = [
                {"$group": {"_id": "$category_name", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            category_stats = list(collection.aggregate(category_pipeline))

            # Recent questions (last 24 hours)
            recent_filter = {
                "created_at": {
                    "$gte": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                }
            }
            recent_questions = collection.count_documents(recent_filter)

            return {
                "total_questions": total_questions,
                "questions_with_embeddings": questions_with_embeddings,
                "embedding_coverage_percentage": (questions_with_embeddings / total_questions * 100) if total_questions > 0 else 0,
                "recent_questions_today": recent_questions,
                "categories": category_stats,
                "collection_name": self.collection_name
            }

        except Exception as e:
            logger.error(f"Error getting question stats: {str(e)}")
            raise

    def get_categories(self) -> List[Dict[str, Any]]:
        """Get all unique categories with question counts"""
        try:
            collection = self.get_collection()

            pipeline = [
                {
                    "$group": {
                        "_id": {
                            "category_id": "$category_id",
                            "category_name": "$category_name"
                        },
                        "question_count": {"$sum": 1},
                        "questions_with_embeddings": {
                            "$sum": {
                                "$cond": [
                                    {"$ne": ["$embedding_vector", None]},
                                    1,
                                    0
                                ]
                            }
                        }
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "category_id": "$_id.category_id",
                        "category_name": "$_id.category_name",
                        "question_count": 1,
                        "questions_with_embeddings": 1,
                        "embedding_coverage": {
                            "$multiply": [
                                {"$divide": ["$questions_with_embeddings", "$question_count"]},
                                100
                            ]
                        }
                    }
                },
                {"$sort": {"question_count": -1}}
            ]

            return list(collection.aggregate(pipeline))

        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            raise

    def create_indexes(self):
        """Create necessary indexes for the collection including vector index"""
        try:
            collection = self.get_collection()

            # Create text index for search
            try:
                collection.create_index([("question_text", "text"), ("category_name", "text")], name="text_search_idx")
                logger.info("Created text search index")
            except Exception as e:
                logger.warning(f"Text index creation failed (may already exist): {str(e)}")

            # Create indexes for common queries
            try:
                collection.create_index("category_id", name="category_id_idx")
                collection.create_index("category_name", name="category_name_idx")
                collection.create_index("created_at", name="created_at_idx")
                logger.info("Created basic indexes")
            except Exception as e:
                logger.warning(f"Basic index creation failed (may already exist): {str(e)}")

            # Create index for embedding field (for existence checks and basic queries)
            try:
                collection.create_index("embedding_vector", name="embedding_vector_idx")
                logger.info("Created embedding vector index")
            except Exception as e:
                logger.warning(f"Embedding vector index creation failed (may already exist): {str(e)}")

            logger.info("Completed index creation for semantic_question collection")

        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
            raise


    def get_index_info(self) -> Dict[str, Any]:
        """Get information about all indexes on the collection"""
        try:
            collection = self.get_collection()

            # Get regular indexes
            regular_indexes = list(collection.list_indexes())

            # Try to get vector search indexes (MongoDB Atlas)
            vector_indexes = []
            try:
                database = collection.database
                search_indexes = database.command("listSearchIndexes", self.collection_name)
                vector_indexes = search_indexes.get("cursor", {}).get("firstBatch", [])
            except Exception:
                pass  # Not MongoDB Atlas or no vector indexes

            return {
                "collection_name": self.collection_name,
                "regular_indexes": regular_indexes,
                "total_regular_indexes": len(regular_indexes),
                "total_vector_indexes": len(vector_indexes)
            }

        except Exception as e:
            logger.error(f"Error getting index info: {str(e)}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the repository including index status"""
        try:
            collection = self.get_collection()

            # Test basic operations
            count = collection.count_documents({})

            # Test connection
            is_connected = self.mongodb.is_connected()

            # Check indexes
            index_info = self.get_index_info()

            return {
                "status": "healthy" if is_connected else "unhealthy",
                "collection_name": self.collection_name,
                "document_count": count,
                "mongodb_connected": is_connected,
                "embedding_service_available": self.embedding_service is not None,
                "regular_indexes_count": index_info['total_regular_indexes']
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "collection_name": self.collection_name,
                "mongodb_connected": False,
                "embedding_service_available": self.embedding_service is not None,
                "vector_search_ready": False
            }
