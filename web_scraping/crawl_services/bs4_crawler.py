import os
import asyncio
import aiohttp
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from searx.engines.utils.logger import get_logger
from web_scraping.crawl_services.base import BaseCrawler
import time
from typing import Optional

class BS4Crawler(BaseCrawler):
    def __init__(self):
        self.logger = get_logger()
    
    async def crawl(self, url: str) -> Optional[str]:
        start = time.monotonic()
        try:
            self.logger.info(f"BS4Crawler: Attempting to crawl {url}")
            
            # Fetch HTML content
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    html = await resp.text()
            
            # Parse HTML and convert to markdown
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove unwanted elements
            for tag in soup(["nav", "footer", "script", "style", "aside", "header"]):
                tag.decompose()
            
            # Find main content
            main = soup.find("main") or soup.find(class_="content") or soup.find("article") or soup.find("body")
            
            # Convert to markdown
            markdown_content = md(str(main), heading_style="ATX")
            
            elapsed = time.monotonic() - start
            self.logger.info(f"BS4Crawler: Successfully crawled {url} in {elapsed:.3f}s")
            
            return markdown_content
            
        except Exception as e:
            elapsed = time.monotonic() - start
            self.logger.error(f"BS4Crawler: Failed to crawl {url} in {elapsed:.3f}s: {str(e)}")
            return None