import requests
import json
from typing import Optional
from os import getenv

import logging

logger: logging.Logger

go_scraper_endpoint: Optional[str] = getenv("GO_SCRAPER_ENDPOINT")

def postprocess_web_search_results(results):
    try:
        urls = [result.get('url') for result in results]

        content_extra = scrape_urls(urls).get("data", [])
        if not content_extra:
            return results
        map_url_content_extra = {
            item['url']: item.get("snippet", "")
            for item in content_extra
        }
        for result in results:
            result["snippet"] = result.get("content", "")
            result["content"] = map_url_content_extra.get(result.get('url'), "")
    except Exception as e:
        logger.error(e, exc_info=True)
    return results

def scrape_urls(urls, endpoint=go_scraper_endpoint):
    payload = json.dumps({
        "urls": urls,
        "timeout": 3,
    })
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(endpoint, headers=headers, data=payload)
        return response.json()
    except Exception as e:
        logger.error(e, exc_info=True)
        return {}
