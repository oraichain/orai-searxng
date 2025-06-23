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
from searx.engines.utils.logger import Logger
from searx.engines.utils.postgres_semantic_servicce import PostgresSemanticService

# engine specific variables
engine_type = 'offline'
paging = True
results_per_page = 20
exact_match_only = False

_searcher: Optional[PostgresSemanticService] = None


# setup logger
logger = Logger(name=__name__, log_to_file=False)


def init(engine_settings: Dict[str, Any]) -> None:
    """Initialize the engine with settings from settings.yml.

    Args:
        engine_settings: Dictionary containing engine configuration settings.

    Raises:
        ValueError: If required settings are missing or invalid.
    """
    global _searcher

    logger.info("Initializing Postgres Semantic Query engine")

    try :
        _searcher = PostgresSemanticService(engine_settings['postgres_semantic_base_url'])
        logger.info("Successfully initialized Postgres Semantic Query engine")
    except Exception as e:
        logger.error("Failed to initialize Postgres Semantic Query engine: %s", str(e))
        raise


# def request(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """Handle search request.

#     Args:
#         query: The search query string.
#         params: Additional search parameters.

#     Returns:
#         List of search results as dictionaries.

#     Note:
#         Returns an empty list if the search fails or no results are found.
#     """
#     logger.debug("Processing query: %s", query)

#     if not _searcher:
#         connect()

#     try:
#         results = _searcher.query_data(query)
#         logger.debug("Found %d results", len(results))
#         return results
#     except Exception as e:
#         logger.error("Error in request: %s", str(e))
#         return []


def search(query: str, params: Dict[str, Any]) -> EngineResults:
    """Perform a vector search query and format results.

    Args:
        query: The search query string.
        params: Additional search parameters.

    Returns:
        EngineResults object containing formatted search results.
    """
    logger.debug("Starting search for query: %s, params: %s", query, params)
    res = EngineResults()

    try:
        results = _searcher.search(query)
        print ("============================================", results)
        if not results or len(results) == 0:
            logger.debug("No results found")
            return res

        processed_results = []
        for idx, result in enumerate(results):
            processed_results.append({f"result {idx}" : result})

        res.add(res.types.LegacyResult(number_of_results=len(processed_results)))

        for row in processed_results:
            if '_id' in row:
                del row['_id']
            kvmap = {str(k): str(v) for k, v in row.items()}
            res.add(res.types.KeyValue(kvmap=kvmap))

        logger.debug("Successfully formatted %d results", len(results))
        return res

    except Exception as e:
        logger.error("Error in search: %s", str(e))
        return res
