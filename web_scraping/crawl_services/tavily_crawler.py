from tavily import AsyncTavilyClient
import os 
from dotenv import load_dotenv
import time
from searx.engines.utils.logger import get_logger
from web_scraping.crawl_services.base import BaseCrawler
load_dotenv()

class TavilyCrawler(BaseCrawler):
    def __init__(self):
        self.logger = get_logger()
        self.client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    async def crawl(self, url: str) -> str:
        start = time.monotonic()
        result = await self.client.extract([url])
        elapsed = time.monotonic() - start
        self.logger.info(f"TavilySearch.extract: url={url} took {elapsed:.3f}s")
        return result['results'][0]['raw_content']
    
    async def search(self, query: str) -> str:
        start = time.monotonic()
        result = await self.client.search(query)
        elapsed = time.monotonic() - start
        self.logger.info(f"TavilySearch.search: query={query} took {elapsed:.3f}s")
        return result['results'][0]['content']


