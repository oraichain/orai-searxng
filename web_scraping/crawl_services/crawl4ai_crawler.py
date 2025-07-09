from crawl4ai import AsyncWebCrawler
from searx.engines.utils.logger import get_logger
from web_scraping.crawl_services.base import BaseCrawler
from typing import Optional
import time

class Crawl4aiCrawler(BaseCrawler):
    def __init__(self):
        self.logger = get_logger()

    async def crawl(self, url: str) -> Optional[str]:
        start = time.monotonic()
        try:
            self.logger.info(f"Crawl4aiPlugin: Attempting extraction for: {url}")
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                if result and result.markdown:
                    self.logger.info(f"Crawl4aiPlugin: Success")
                    return result.markdown.strip()
        except Exception as e:
            self.logger.error(f"Crawl4aiPlugin: Failed for {url}: {str(e)}")
        return None
