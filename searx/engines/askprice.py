# SPDX-License-Identifier: AGPL-3.0-or-later
"""Price Asking engine for semantic search capabilities.

This engine provides token price functionality using Coinmarket cap api capabilities
combined with OpenAI LLM.

Configuration
=============

To use this engine, you need to configure the following settings:

.. code:: yaml

  - name: askprice
    engine: askprice
    shortcut: asp
    disabled: false
    timeout: 10

Required dependencies:
- openai
- python-dotenv
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from searx.result_types import EngineResults
from searx.engines.utils.llm import LLM
from searx.engines.utils.schema import Message

# engine specific variables
engine_type = 'offline'
openai_api_key: str 
cmc_api_key: str
_llm: Optional[LLM] = None
model: Optional[str] = None
logger = logging.getLogger('searx.engines.askprice')

def init(engine_settings: Dict[str, Any]) -> None:
    """Initialize the engine with settings from settings.yml.
    
    Args:
        engine_settings: Dictionary containing engine configuration settings.
        
    Raises:
        ValueError: If required settings are missing or invalid.
    """
    global openai_api_key, cmc_api_key
    required_settings = ['openai_api_key' , 'cmc_api_key']
    for setting in required_settings:
        if setting not in engine_settings:
            raise ValueError(f'Missing required setting: {setting}')
        if not engine_settings[setting]:
            raise ValueError(f'Empty value for required setting: {setting}')
    logger.info("Initializing ask_price engine with settings: %s", engine_settings)
    connect()


def connect() -> None:
    """Initialize the ask price connection.
    
    This function creates a new LLM instance with the configured 
    OpenAI API key.
    
    Raises:
        Exception: If connection fails.
    """
    global _llm 
    
    try:
        if model is None:
            raise ValueError("Model is not set. Please configure the model in settings.")

        _llm = LLM(
            model=model,
            api_key = openai_api_key,
        )

        logger.info("Successfully initialized OpenAI model: %s", _llm.model)
    except Exception as e:
        logger.error("Failed to initialize SearchService: %s", str(e))
        raise

def request(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle search request.
    
    Args:
        query: The search query string.
        
    Returns:
        List of search results as dictionaries.
        
    Note:
        Returns an empty list if the search fails or no results are found.
    """
    logger.debug("Processing query: %s", query)
    
    if not _llm:
        connect()
        
    try:
        message = [Message.user_message(query)]
        results = _llm.function_calling(message, cmc_api_key=cmc_api_key)
        results = _llm.parse_tool_outputs(results)
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
