import concurrent.futures
import asyncio

def execute_multithreading_functions(functions, timeout=300):
    """
    Execute multiple functions in parallel using ThreadPoolExecutor.
    Each function is a dict {'fn': function, 'args': dict}.
    Return a list of results in the same order.
    If timeout, raise Exception.
    If a function fails, return the exception object at that position.
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

async def execute_async_functions(functions, timeout=300):
    """
    Execute multiple async functions in parallel.
    Each element in functions is a dict {'fn': async function, 'args': dict}.
    Return a list of results in the same order.
    If timeout, raise Exception.
    If a function fails, return the exception object at that position.
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


