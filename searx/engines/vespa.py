# SPDX-License-Identifier: AGPL-3.0-or-later
"""Price Asking engine for semantic search capabilities. (MOCK IMPLEMENTATION)

This engine, named 'vespa', provides a mocked token price functionality.
It does not connect to any external APIs like Coinmarket Cap or OpenAI. The returned
data is hardcoded for testing and demonstration purposes.

Configuration
=============

To use this engine, you can use a minimal configuration as no API keys
are required.

.. code:: yaml

  - name: vespa # Updated engine name in configuration
    engine: vespa
    shortcut: mte # Example shortcut
    disabled: false
    timeout: 10

Required dependencies:
- None (all external dependencies removed for this mock version)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from searx.result_types import EngineResults
from searx.engines.utils.llm import LLM




# engine specific variables
engine_type = 'offline'
openai_api_key: str
_llm: Optional[LLM] = None
model: Optional[str] = None


# Updated logger name to reflect the new engine name
logger = logging.getLogger('searx.engines.vespa')


def init(engine_settings: Dict[str, Any]) -> None:
    """Initialize the vespa.

    Args:
        engine_settings: Dictionary containing engine configuration settings (ignored).
    """
    # No API keys or external connections are needed for this mock implementation.
    global openai_api_key, model
    required_settings = ['openai_api_key' , 'model']
    for setting in required_settings:
        if setting not in engine_settings:
            raise ValueError(f'Missing required setting: {setting}')
        if not engine_settings[setting]:
            raise ValueError(f'Empty value for required setting: {setting}')
    connect()

    logger.info("Initializing vespa. All results will be hardcoded.")


def connect() -> None:
    """Initialize the LLM connection.

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


def classify(query: str) -> str:
    """Classify the query into a category."""
    # This function's logic is simple and can be kept as is.
    return "price"


def request(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:

    global _llm
    """Handle search request by returning mocked data for vespa.

    This function ignores the input query and returns a static, predefined
    list of results.

    Args:
        query: The search query string (ignored).
        params: Additional search parameters (ignored).

    Returns:
        A list of hardcoded search results as dictionaries.
    """
    logger.debug("Processing query '%s' with mock data from vespa.", query)

    # Return a copy of the mock data to prevent any potential modification
    # of the global variable.

    logger.info("\n\n\n The query: %s  \n\n\n", query)
    if _llm is None:
        raise ValueError("LLM not initialized. Please call init() first.")


    prompt = """
You are a DeFI expert. Your task is classify this query into one of below query categories:
- `potential_meme_token`: For DeFI related query ask for meme token to invest
- `stablecoin_yield_farming`: For DeFI related query ask for stablecoin yield farming
- `whale_analysis`: For DeFI related query ask for whale analysis
- `other`: For all other DeFI related query or non DeFI query


User's query = $$$QUERY$$$

Your task is return type of user's query in json format as below, without explain anything more
```json
{
    "query_type": "potential_meme_token" or "stablecoin_yield_farming" or "whale_analysis" or "other"
}
```
"""


    prompt = prompt.replace("$$$QUERY$$", query)
    json_response = _llm.gen_structure_output(
        prompt=prompt,
        default_response={"query_type": "other"}
    )
    data = [json_response]
    return data


def search(query: str, params: Dict[str, Any]) -> EngineResults:
    """Perform a mock search query using vespa and format results.

    Args:
        query: The search query string.
        params: Additional search parameters.

    Returns:
        EngineResults object containing formatted mock search results.
    """
    logger.debug("Starting mock search for query: %s (vespa)", query)
    res = EngineResults()

    try:
        # The 'request' function now returns our hardcoded mock data.
        results = request(query, params)
        if not results:
            logger.debug("No mock results found or configured for vespa.")
            return res

        res.add(res.types.LegacyResult(number_of_results=len(results)))

        # The original formatting logic is preserved and now works on the mock data.
        for row in results:
            kvmap = {str(k): str(v) for k, v in row.items()}
            res.add(res.types.KeyValue(kvmap=kvmap))

        logger.debug("Successfully formatted %d mock results from vespa", len(results))
        return res

    except Exception as e:
        # Keep error handling as a good practice.
        logger.error("Error in vespa search processing: %s", str(e))
        return res
