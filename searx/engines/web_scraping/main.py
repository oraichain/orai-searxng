from searx.engines.web_scraping.tavily_search import TavilySearch
from searx.engines.web_scraping.crawl4ai_search import Crawl4aiSearch
from searx.engines.web_scraping.jina_search import JinaSearch
from searx.engines.web_scraping.benchmark import benchmark_bulk_extract
import asyncio
import json
from pathlib import Path
import time


tavily_search = TavilySearch()
crawl4ai_search = Crawl4aiSearch()
jina_search = JinaSearch()

async def main():
    query = "python" 
    path = Path("searx/engines/web_scraping/output")
    path.mkdir(parents=True, exist_ok=True)

    response_path = path / "response.json"
    if not response_path.exists():
        print(f"{response_path} not found. Please run search and save results to response.json first.")
        return

    with open(response_path, "r") as f:
        response_data = json.load(f)
    urls = [item["url"] for item in response_data.get("results", []) if "url" in item]
    print(f"Found {len(urls)} urls to crawl with JinaSearch.")

    # Benchmark crawl all urls with JinaSearch
    result = await benchmark_bulk_extract(jina_search, urls)
    elapsed = result["elapsed"]
    print(f"Bulk JinaSearch crawl for {len(urls)} urls took {elapsed:.3f}s")

    with open(path / "jina_bulk_crawl.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    asyncio.run(main())