from typing import Optional
from .mongodb import MongoDB
from .data_repository import DataRepository
from .defi_semantic_question_repository import DefiSemanticQuestionRepository
from .embedding import EmbeddingService
from .logger import get_logger

from typing import Tuple, List, Dict, Any



def truncate_content(content: str, max_chars: int | None = None) -> str:
    if max_chars is None or len(content) <= max_chars or max_chars < 0:
        return content
    half = max_chars // 2
    return content[:half] + '\n[... Observation truncated due to length ...]\n' + content[-half:]

class SearchService:
    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        collection_vector: Optional[str] = None
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
                embedding_service=self.embedding_service,
                collection_vector=collection_vector
            )
            self.data_repo = DataRepository(mongodb=self.mongo_db, collection_name=collection_name)

        except Exception as e:
            self.logger.exception("Error during SearchService initialization: %s", str(e))
            raise


    @staticmethod
    def post_process_output(
        input_json: List[Dict[str, Any]],
        max_token_chars: int = 12000
    ) -> List[Dict[str, Any]]:
        # markdown_blocks = []

        for item in input_json:
            value = item.get("value", {})
            num_part = len(value.keys())
            # Build markdown parts dynamically based on value keys
            parts = []
            max_chars = max_token_chars // len(input_json) // num_part
            print(max_chars)
            for key, content in value.items():
                # Truncate content
                truncated_content = truncate_content(
                    content = str(content),
                    max_chars = max_chars
                )
                value[key] = truncated_content

                # if truncated_content:
                #     # Convert key to title case for Markdown heading
                #     title = ' '.join(word.capitalize() for word in key.split())
                #     parts.append(f"## {title}\n{truncated_content}")

            # # Join parts into a single markdown block
            # markdown_text = "\n\n".join(parts)
            # if markdown_text:
            #     markdown_blocks.append(markdown_text)

        # return markdown_blocks
        return input_json

    def query_data(self, query: str, max_chars_output: int = 20000) -> List[Dict[str, Any]]:
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
            result_data = self.post_process_output(result_data, max_chars_output)
            return result_data
        except Exception as e:
            self.logger.exception("Error during data query: %s", str(e))
            return []
