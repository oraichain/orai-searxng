import requests
import json
from openai import OpenAI
from typing import Any, List, Dict, Set, Tuple
from openai.types.chat import ChatCompletionToolParam

def fetch_from_coinmarketcap(
    symbols: Set[str],
    cmc_api_key: str,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Fetch prices from CoinMarketCap API for a set of symbols."""
    api_key = cmc_api_key
    if not api_key:
        raise ValueError("CMC_API_KEY environment variable not set.")
        
    base_url = 'https://pro-api.coinmarketcap.com/v1'
    
    results = []
    failed_symbols = symbols.copy()
    
    if not symbols:
        return results, failed_symbols

    try:
        url = f"{base_url}/cryptocurrency/quotes/latest"
        # CoinMarketCap API is case-insensitive but often uses uppercase
        params = {"symbol": ",".join(s.upper() for s in symbols)}
        headers = {"X-CMC_PRO_API_KEY": api_key}

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
            
        data = response.json()
        if not isinstance(data, dict) or "data" not in data:
            raise Exception("Invalid response format from CoinMarketCap")

        for symbol in list(failed_symbols):
            token_data = data["data"].get(symbol.upper())
            if (
                token_data
                and "quote" in token_data
                and "USD" in token_data["quote"]
            ):
                current_price = token_data["quote"]["USD"]["price"]
                results.append(
                    {
                        "symbol": symbol.upper(),
                        "name": token_data["name"],
                        "price": current_price,
                    }
                )
                failed_symbols.remove(symbol)
                
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as e:
        print(f"Failed to fetch from CoinMarketCap: {str(e)}")
        
    return results, failed_symbols

function_mapping = {
    "fetch_from_coinmarketcap": fetch_from_coinmarketcap,
}

tools_schema: List[ChatCompletionToolParam] = [{
    "type": "function",
    "function": {
        "name": "fetch_from_coinmarketcap",
        "description": "Get the latest price in USD for a list of cryptocurrency symbols.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of cryptocurrency symbols (e.g., ['BTC', 'ETH']).",
                }
            },
            "required": ["symbols"],
        },
    }
}]