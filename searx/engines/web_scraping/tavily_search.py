from tavily import AsyncTavilyClient
import os 
from dotenv import load_dotenv
import time
from searx.engines.utils.logger import get_logger
load_dotenv()

class TavilySearch():
    def __init__(self):
        self.logger = get_logger()
        self.client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    async def crawl(self, url: str, **kwargs) -> str:
        start = time.monotonic()
        res = await self.client.extract([url])
        elapsed = time.monotonic() - start
        self.logger.info(f"TavilySearch.extract: url={url} took {elapsed:.3f}s")
        return res
    
    async def search(self, query: str, **kwargs) -> str:
        start = time.monotonic()
        res = await self.client.search(query)
        elapsed = time.monotonic() - start
        self.logger.info(f"TavilySearch.search: query={query} took {elapsed:.3f}s")
        return res


