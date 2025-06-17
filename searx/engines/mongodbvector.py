# SPDX-License-Identifier: AGPL-3.0-or-later
"""MongoDB Vector Search engine for semantic search capabilities.

This engine provides vector search functionality using MongoDB's vector search capabilities
combined with OpenAI embeddings.

Configuration
=============

To use this engine, you need to configure the following settings:

.. code:: yaml

  - name: mongodbvector
    engine: mongodbvector
    shortcut: mdv
    mongodb_uri: 'mongodb://user:pass@host:port/db'
    openai_api_key: 'your-openai-api-key'
    results_per_page: 20
    database: 'your_database'
    collection: 'your_collection'
    collection_vector: 'your_vector_collection'

Required dependencies:
- pymongo
- openai
- python-dotenv
- numpy

Example
=======

Below is an example configuration:

.. code:: yaml

  - name: mongodbvector
    engine: mongodbvector
    shortcut: mdv
    mongodb_uri: 'mongodb://localhost:27017'
    openai_api_key: 'sk-...'
    results_per_page: 20
    database: 'semantic_search'
    collection: 'documents'
    collection_vector: 'vector_documents'

"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from searx.engines.utils.search_service import SearchService
from searx.result_types import EngineResults

# engine specific variables
engine_type = 'offline'
paging = True
results_per_page = 20
exact_match_only = False

# mongodb connection variables
mongodb_uri: Optional[str] = None
openai_api_key: Optional[str] = None
database: Optional[str] = None
collection: Optional[str] = None
collection_vector: Optional[str] = None
_searcher: Optional[SearchService] = None

# setup logger
logger = logging.getLogger('searx.engines.mongodbvector')


def init(engine_settings: Dict[str, Any]) -> None:
    """Initialize the engine with settings from settings.yml.
    
    Args:
        engine_settings: Dictionary containing engine configuration settings.
        
    Raises:
        ValueError: If required settings are missing or invalid.
    """
    global mongodb_uri, openai_api_key, database, collection, collection_vector  # pylint: disable=global-statement
    
    # Validate required settings
    required_settings = ['mongodb_uri', 'openai_api_key', 'database', 'collection', 'collection_vector']
    for setting in required_settings:
        if setting not in engine_settings:
            raise ValueError(f'Missing required setting: {setting}')
        if not engine_settings[setting]:
            raise ValueError(f'Empty value for required setting: {setting}')
    
    mongodb_uri = engine_settings['mongodb_uri']
    openai_api_key = engine_settings['openai_api_key']
    database = engine_settings['database']
    collection = engine_settings['collection']
    collection_vector = engine_settings['collection_vector']
    
    logger.info("Initializing MongoDB Vector Search engine with database: %s, collection: %s", 
                database, collection)
    
    connect()


def connect() -> None:
    """Initialize the search service connection.
    
    This function creates a new SearchService instance with the configured
    MongoDB URI and OpenAI API key.
    
    Raises:
        Exception: If connection fails.
    """
    global _searcher  # pylint: disable=global-statement
    
    try:
        _searcher = SearchService(
            mongodb_uri=mongodb_uri,
            openai_api_key=openai_api_key,
            collection_name=collection,
            collection_vector=collection_vector
        )
        logger.info("Successfully connected to MongoDB and initialized SearchService")
    except Exception as e:
        logger.error("Failed to initialize SearchService: %s", str(e))
        raise


def request(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle search request.
    
    Args:
        query: The search query string.
        params: Additional search parameters.
        
    Returns:
        List of search results as dictionaries.
        
    Note:
        Returns an empty list if the search fails or no results are found.
    """
    logger.debug("Processing query: %s", query)
    
    if not _searcher:
        connect()
        
    try:
        results = _searcher.query_data(query)
        logger.debug("Found %d results", len(results))
        return results
    except Exception as e:
        logger.error("Error in request: %s", str(e))
        return []


def search(query: str, params: Dict[str, Any]) -> EngineResults:
    """Perform a vector search query and format results.
    
    Args:
        query: The search query string.
        params: Additional search parameters.
        
    Returns:
        EngineResults object containing formatted search results.
    """
    logger.debug("Starting search for query: %s", query)
    res = EngineResults()
    
    try:
        results = request(query, params)
        if not results:
            logger.debug("No results found")
            return res
            
        res.add(res.types.LegacyResult(number_of_results=len(results)))
        
        for row in results:
            if '_id' in row:
                del row['_id']
            kvmap = {str(k): str(v) for k, v in row.items()}
            res.add(res.types.KeyValue(kvmap=kvmap))
            
        logger.debug("Successfully formatted %d results", len(results))
        return res
        
    except Exception as e:
        logger.error("Error in search: %s", str(e))
        return res
