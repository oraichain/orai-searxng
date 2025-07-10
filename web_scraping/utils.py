import concurrent.futures
import asyncio
from typing import List, Dict, Any
from pathlib import Path
import json
import requests
import httpx

def execute_multithreading_functions(functions: List[Dict[str, Any]], timeout: int = 300) -> List[Any]:
    """
    Execute multiple functions in parallel using ThreadPoolExecutor.

    Args:
        functions: List of dicts, each containing a function and its arguments.
        timeout: Timeout in seconds.

    Returns:
        List of results in the same order.
    """
    try:
        results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for function in functions:
                results.append(executor.submit(function["fn"], **function["args"]))

        return [result.result(timeout=timeout) for result in results]

    except TimeoutError:
        raise Exception("Function execution timed out")
    except Exception as e:
        raise Exception(f"Error executing multithreading functions: {e}")

async def execute_async_functions(functions, timeout=30):
    """
    Execute multiple async functions in parallel.

    Args:
        functions: List of dicts, each containing an async function and its arguments.
        timeout: Timeout in seconds.

    Returns:
        List of results in the same order.

    Raises:
        Exception: If the function execution times out.
    """
    async def run_fn(fn, args):
        try:
            return await fn(**args)
        except Exception as e:
            return e

    tasks = [run_fn(f["fn"], f.get("args", {})) for f in functions]
    try:
        return await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
    except asyncio.TimeoutError:
        raise Exception("Async function execution timed out")


def save_to_json_file(data: Any, file_name: str) -> None:
    """
    Save data to a file.

    Args:
        data: Data to save.
        file_name: Name of the file to save.
    """
    path = Path("searx/engines/web_scraping/output")
    path.mkdir(parents=True, exist_ok=True)
    path = path / file_name
    with open(path, "w") as f:
        json.dump(data, f)

async def call_searxng_api(query: str) -> Dict[str, Any]:
    url = f"http://148.113.35.59:8666/search?format=json&q={query}&engines=google"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    return response.json()