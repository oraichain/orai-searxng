import requests
import json
from pathlib import Path
import time
import asyncio
from typing import List, Any, Dict

def call_searxng_api(query: str) -> List[Dict[str, Any]]:
    """
    Call the searxng API to search for a query.
    """
    path = Path("searx/engines/web_scraping/output")
    path.mkdir(parents=True, exist_ok=True)
    url = f"http://148.113.35.59:8666/search?format=json&q={query}"
    response = requests.get(url)
    path = Path("searx/engines/web_scraping/output/response.json")
    with open(path, "w") as f:
        json.dump(response.json(), f)
    return response.json()


def _get_bulk_crawl_fn(tool):
    # Use crawl_multiple_urls if available, otherwise create a custom bulk crawl function
    if hasattr(tool, 'crawl_multiple_urls') and callable(getattr(tool, 'crawl_multiple_urls')):
        return tool.crawl_multiple_urls
    async def bulk_crawl(urls: List[str]):
        sem = asyncio.Semaphore(10)
        async def sem_crawl(url):
            async with sem:
                return await tool.crawl(url)
        tasks = [sem_crawl(url) for url in urls]
        return await asyncio.gather(*tasks)
    return bulk_crawl

async def benchmark_bulk_extract(tool, urls: List[str]) -> Dict[str, Any]:
    """
    Run async crawl for all urls in parallel using tool, measure total time, return dict {'elapsed': elapsed, 'results': results}
    """
    start = time.monotonic()
    bulk_crawl = _get_bulk_crawl_fn(tool)
    results = await bulk_crawl(urls)
    elapsed = time.monotonic() - start
    return {'elapsed': elapsed, 'results': results}


if __name__ == "__main__":
    print(call_searxng_api("python"))