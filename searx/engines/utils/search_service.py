import os
import json
from typing import Optional
from dotenv import load_dotenv
from .mongodb import MongoDB
from .data_repository import DataRepository
from .defi_semantic_question_repository import DefiSemanticQuestionRepository
from .embedding import EmbeddingService
from .logger import get_logger

load_dotenv()

MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class SearchService:
    def __init__(
        self,
        mongodb_uri: Optional[str] = MONGO_CONNECTION_STRING,
        openai_api_key: Optional[str] = OPENAI_API_KEY
    ):
        self.logger = get_logger()
        try:
            if mongodb_uri is None:
                raise ValueError("MONGO_CONNECTION_STRING is not set")
            self.mongo_db = MongoDB(mongodb_uri)
            self.mongo_db.connect()
            self.logger.info("MongoDB connection established at URI: %s", mongodb_uri)

            self.embedding_service = EmbeddingService(
                model="text-embedding-3-large",
                api_key=openai_api_key
            )
            self.question_repo = DefiSemanticQuestionRepository(
                mongodb=self.mongo_db,
                embedding_service=self.embedding_service
            )
            self.data_repo = DataRepository(mongodb=self.mongo_db)

        except Exception as e:
            self.logger.exception("Error during SearchService initialization: %s", str(e))
            raise

    def query_data(self, query: str):
        self.logger.info("Starting semantic search for query: '%s'", query)
        try:
            similar_questions = self.question_repo.find_similar_questions(
                query_text=query,
                limit=3,
                similarity_threshold=0.5
            )

            self.logger.info("Found %d similar questions", len(similar_questions))
            filtered_category_ids = set(q['category_id'] for q in similar_questions)

            result_data = []
            for category_id in filtered_category_ids:
                data = self.data_repo.query_by_category_id(category_id=category_id)
                if data:
                    result_data.append(data[0])

            return result_data
        except Exception as e:
            self.logger.exception("Error during data query: %s", str(e))
            return []
