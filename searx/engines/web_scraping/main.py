from searx.engines.web_scraping.tavily_search import TavilyCrawler
from searx.engines.web_scraping.crawl4ai_search import Crawl4aiCrawler
from searx.engines.web_scraping.jina_search import JinaCrawler
from searx.engines.web_scraping.utils import call_searxng_api
import asyncio
import json
from pathlib import Path
import time
from searx.engines.web_scraping.utils import save_to_json_file
from searx.engines.web_scraping.crawler import AsyncWebCrawler

plugin = TavilyCrawler()

async def main():
    query = "Bitcoin price"
    
    # Gọi SearXNG API
    response_data = call_searxng_api(query)

    # Lấy danh sách URL từ kết quả
    urls = [item["url"] for item in response_data.get("results", []) if "url" in item]
    print(f"[INFO] Found {len(urls)} URLs from SearXNG for query: '{query}'")

    # Nếu không có URL, dừng lại
    if not urls:
        print("[WARN] No URLs found in SearXNG response.")
        return

    # Crawler
    crawler = AsyncWebCrawler(plugin=plugin, max_concurrency=20)
    results = await crawler.crawl_many(urls)

    for url, content in zip(urls, results):
        print(f"\n--- {url} ---")
        print(content[:500] if content else "None, ❌ Failed to extract content")


if __name__ == "__main__":
    asyncio.run(main())