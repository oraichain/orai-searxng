from typing import Optional
from openai import OpenAI, OpenAIError
from typing import List, Union
import logging


logger = logging.getLogger(__name__)
class EmbeddingService:
    """Service for generating text embeddings using OpenAI-compatible API"""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize the embedding service

        Args:
            model: The embedding model to use (default: text-embedding-3-small)
        """
        self.client = OpenAI(
            api_key=api_key
        )
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            OpenAIError: If the API call fails
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except OpenAIError as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            OpenAIError: If the API call fails
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except OpenAIError as e:
            logger.error(f"Error generating embeddings for texts: {str(e)}")
            raise

    def embed_document(self, document: Union[str, dict]) -> List[float]:
        """
        Generate embedding for a document (can be string or dict)

        Args:
            document: The document to embed (string or dict)

        Returns:
            List of floats representing the embedding vector
        """
        if isinstance(document, dict):
            # Convert dict to string representation for embedding
            text = self._dict_to_text(document)
        else:
            text = str(document)

        return self.embed_text(text)

    def _dict_to_text(self, data: dict) -> str:
        """
        Convert dictionary to text representation for embedding

        Args:
            data: Dictionary to convert

        Returns:
            String representation of the dictionary
        """
        def extract_text_from_dict(d, prefix=""):
            text_parts = []
            for key, value in d.items():
                if isinstance(value, dict):
                    text_parts.extend(extract_text_from_dict(value, f"{prefix}{key}."))
                elif isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            text_parts.extend(extract_text_from_dict(item, f"{prefix}{key}[{i}]."))
                        else:
                            text_parts.append(f"{prefix}{key}[{i}]: {str(item)}")
                else:
                    text_parts.append(f"{prefix}{key}: {str(value)}")
            return text_parts

        text_parts = extract_text_from_dict(data)
        return " | ".join(text_parts)
