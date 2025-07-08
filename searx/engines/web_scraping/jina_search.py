import httpx

from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import AsyncOpenAI
from dotenv import load_dotenv
import requests
from typing import Optional, Any
import time
from searx.engines.utils.logger import get_logger

logger = get_logger()
load_dotenv()

class JINASettings(BaseSettings):
    jina_key: str 
    jina_searcher_endpoint: str
    base_url: str
    api_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore"  # Cho phép các biến môi trường thừa
    )

jina_settings = JINASettings()

client = AsyncOpenAI(api_key=jina_settings.api_key, base_url=jina_settings.base_url)


def extract_with_jina(url: str) -> Optional[str]:
    """Extract text using Jina AI reader service."""
    try:
        logger.info(f"Attempting Jina AI extraction for: {url}")
        url = f"https://r.jina.ai/{url}"
        payload = {}
        headers = {
            'Authorization': f'Bearer {jina_settings.jina_key}',
            'X-Engine': 'cf-browser-rendering',
            'X-Return-Format': 'markdown'
        }
        response = requests.request("GET", url, headers=headers, data=payload)
        if response.status_code == 200 and response.text.strip():
            return response.text.strip()
        else:
            logger.warning(f"Jina AI: Request failed with status {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"Jina AI extraction failed for {url}: {str(e)}")
        return None

async def search_web_content_based_on_keywords(keywords, top_k=5) -> str:
    """Search the web and get SERP

    Args:
        keywords (str): The keyword to search on the web (required)
        top_k (int): The number of top results to return

    Returns:
        str: The returned content
    """
    try:
        keywords = keywords.replace(" ", "+")

        url = f"{jina_settings.jina_searcher_endpoint}"
        params = {
            "q": keywords,
            "num": int(top_k),
        }
        logger.info(f"JinaSearch.search: endpoint={url} params={params}")
        headers = {
            'X-Respond-With': 'no-content',
            "Authorization": f"Bearer {jina_settings.jina_key}"
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, params=params)
        logger.info(f"JinaSearch.search: Response: {response.text}")
    except Exception as e:
        logger.error(f"Error fetching SERP content by Jina API: {e}")
        return f"Error fetching SERP content by Jina API: {e}"
    try:
        if response.json().get('code', 200) != 200:
            logger.warning(f"JinaSearch.search: API error: {response.json()}")
            return f"Error fetching SERP content by Jina API: {response.json()}"
    except Exception:
        pass
    return response.text

class JinaSearch:
    def __init__(self):
        self.logger = get_logger()

    async def crawl(self, url: str, **kwargs) -> Optional[str]:
        start = time.monotonic()
        try:
            # extract_with_jina là sync, nên chạy trong executor
            import asyncio
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, extract_with_jina, url)
            elapsed = time.monotonic() - start
            if content:
                self.logger.info(f"JinaSearch.crawl: Success, took {elapsed:.3f}s")
                return content
            else:
                self.logger.warning(f"JinaSearch.crawl: No content found, took {elapsed:.3f}s")
                return None
        except Exception as e:
            elapsed = time.monotonic() - start
            self.logger.error(f"JinaSearch.crawl: Failed for {url}: {str(e)}, took {elapsed:.3f}s")
            return None

    async def search(self, query: str, top_k: int = 5, **kwargs) -> Any:
        start = time.monotonic()
        try:
            result = await search_web_content_based_on_keywords(query, top_k=top_k)
            elapsed = time.monotonic() - start
            self.logger.info(f"JinaSearch.search: query={query} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.monotonic() - start
            self.logger.error(f"JinaSearch.search: Failed for {query}: {str(e)}, took {elapsed:.3f}s")
            return None

jina_search = JinaSearch()
