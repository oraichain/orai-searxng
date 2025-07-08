from crawl4ai import AsyncWebCrawler
from typing import Optional
import time
from searx.engines.utils.logger import get_logger

class Crawl4aiSearch():
    def __init__(self):
        self.logger = get_logger()

    async def crawl(self, url: str, **kwargs) -> Optional[str]:
        start = time.monotonic()
        try:
            self.logger.info(f"Crawl4aiSearch.extract: Attempting extraction for: {url}")
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                elapsed = time.monotonic() - start
                if result and result.markdown:
                    self.logger.info(f"Crawl4aiSearch.extract: Success, took {elapsed:.3f}s")
                    return result.markdown.strip()
                else:
                    self.logger.warning(f"Crawl4aiSearch.extract: No markdown content found, took {elapsed:.3f}s")
                    return None
        except Exception as e:
            elapsed = time.monotonic() - start
            self.logger.error(f"Crawl4aiSearch.extract: Failed for {url}: {str(e)}, took {elapsed:.3f}s")
            return None