from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from web_scraping.crawl_services.tavily_crawler import TavilyCrawler
from web_scraping.crawl_services.crawl4ai_crawler import Crawl4aiCrawler
from web_scraping.crawl_services.jina_crawler import JinaCrawler
from web_scraping.utils import call_searxng_api
from web_scraping.crawl_services.base import AsyncWebCrawler
from web_scraping.models import ScrapingRequest, ScrapedItem, SearchRequest, SearchResponse

app = FastAPI(title="Scraping Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

plugin = TavilyCrawler()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/crawl", response_model=List[ScrapedItem])
async def crawl_api(request: ScrapingRequest):
    crawler = AsyncWebCrawler(plugin=plugin, max_concurrency=10)
    results = await crawler.crawl_many(request.urls)
    return [{"url": url, "content": content} for url, content in zip(request.urls, results)]

@app.post("/search", response_model=SearchResponse)
async def search_api(request: SearchRequest):
    # Call SearXNG API
    response_data = call_searxng_api(request.query)

    # Extract URLs
    urls = [item["url"] for item in response_data.get("results", []) if "url" in item]

    # Crawl content
    if urls:
        crawler = AsyncWebCrawler(plugin=plugin, max_concurrency=40)
        contents = await crawler.crawl_many(urls)
        # Update content field
        for item, content in zip(response_data["results"], contents):
            if content:  # only replace if crawl was successful
                item["content"] = content

    return SearchResponse(results=response_data.get("results", []))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
