import requests
from searx.engines.utils.logger import Logger

logger = Logger(name=__name__, log_to_file=False)

class PostgresSemanticService:
    def __init__(self, postgres_semantic_base_url: str):
        self.service_base_url = postgres_semantic_base_url

    def search(self, query: str) -> list[dict]:
        try :
            search_url = f"{self.service_base_url}/query-with-user-question"
            body = {
                "question": query
            }
            response = requests.post(search_url, json=body)
            if (response.status_code == 200):
                return response.json()["response"]
            else:
                logger.error("Error in PostgresSemanticService.search", response.text)
                return []
        except Exception as e:
            logger.error("Error in PostgresSemanticService.search", e)
            return []