import asyncio
from typing import List, Optional
from searx.engines.utils.logger import get_logger
from abc import ABC, abstractmethod
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
    after_log,
    RetryError,
    stop_after_delay
)
import httpx 

def is_empty_result(result: Optional[str]) -> bool:
    return result is None

class BaseCrawler(ABC):
    @abstractmethod
    async def crawl(self, url: str) -> Optional[str]:
        pass

class AsyncWebCrawler:
    def __init__(self, plugin: BaseCrawler, max_concurrency: int = 40, timeout: int = 3):
        self.plugin = plugin
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.logger = get_logger()
        self.timeout = timeout

    @retry(
        retry=(
            retry_if_result(is_empty_result) |
            retry_if_exception_type((IndexError, httpx.RequestError, httpx.TimeoutException, ValueError))
        ),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=( stop_after_attempt(2) | stop_after_delay(3) ),
        after=after_log(get_logger().logger, log_level=logging.WARNING),
        reraise=True
    )
    async def _crawl_one_with_retry(self, url: str) -> Optional[str]:
        self.logger.info(f"[AsyncWebCrawler] Crawling {url}")
        try:
            result = await asyncio.wait_for(self.plugin.crawl(url), timeout=self.timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"[AsyncWebCrawler] Timeout (>{self.timeout}s): {url}")
            raise httpx.TimeoutException(f"Timeout while crawling {url}")
        
        if result:
            self.logger.info(f"[AsyncWebCrawler] Success: {url}")
        else:
            self.logger.warning(f"[AsyncWebCrawler] Empty result: {url}")
        return result

    async def _crawl_one(self, url: str) -> Optional[str]:
        async with self.semaphore:
            try:
                return await self._crawl_one_with_retry(url)
            except RetryError as e:
                self.logger.error(f"[AsyncWebCrawler] Retry failed for {url}: {e}")
                return None
            except Exception as e:
                self.logger.error(f"[AsyncWebCrawler] Unexpected error crawling {url}: {e}")
                return None

    async def crawl_many(self, urls: List[str]) -> List[Optional[str]]:
        tasks = [self._crawl_one(url) for url in urls]
        return await asyncio.gather(*tasks)