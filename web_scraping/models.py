from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ScrapingRequest(BaseModel):
    urls: List[str]

class ScrapedItem(BaseModel):
    url: str
    content: Optional[str] = None

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
