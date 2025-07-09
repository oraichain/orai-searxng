from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi import Request
import time
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("uvicorn.error")

class ScrapingRequest(BaseModel):
    urls: List[str]

class ScrapedItem(BaseModel):
    url: str
    content: Optional[str] = None

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

class TimerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        logger.info(f"{request.method} {request.url.path} completed in {duration:.3f}s")
        return response