import os
import json
import base64
from typing import List, Dict, Any, Generator, Optional
from groq import Groq
from openai import OpenAI, OpenAIError
from google import genai
from google.genai import types


def create_crypto_summary_prompt(user_question: str, web_content: str) -> str:
    """
    Create a prompt for summarizing crypto-related web content.

    Args:
        user_question: The user's crypto-related question
        web_content: The web page content to analyze

    Returns:
        Formatted prompt string
    """
    prompt_template = """
You are given:

* A **user question** (Mostly crypto-related question e.g., "Latest news on Solana?")
* The **content of a web page** that was extracted (news article, blog, announcement, etc.)

**Your task** is to analyze the content and **summarize the most important and relevant information** and answer user:

* **5 key bullet points**
* Each point should be **concise, factual, and focused on notable developments or insights**
* Keep the output **as short as possible**, but **retain all critical information** that helps answer the user's question
* Ignore promotional, irrelevant, or generic content

### Question: {user_question}

### Web content:
{web_content}

**Format:**

```
* Point 1
* Point 2
* Point 3
* Point 4
* Point 5

```

---
"""

    return prompt_template.format(
        user_question=user_question,
        web_content=web_content
    )


def generate_with_groq(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Generate response using Groq API.

    Args:
        prompt: The prompt to send to the model
        api_key: Groq API key (defaults to environment variable)

    Returns:
        Generated response text
    """
    client = Groq(
        api_key=api_key or os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        stream=False,
    )

    return chat_completion.choices[0].message.content


def generate_with_sambanova(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Generate response using SambaNova API.

    Args:
        prompt: The prompt to send to the model
        api_key: SambaNova API key (defaults to environment variable)

    Returns:
        Generated response text
    """
    client = OpenAI(
        api_key=api_key or os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )

    response = client.chat.completions.create(
        model="Meta-Llama-3.3-70B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        top_p=0.1
    )

    return response.choices[0].message.content


def generate_with_gemini(prompt: str, api_key: Optional[str] = None) -> None:
    """
    Generate response using Google Gemini API (streams output).

    Args:
        prompt: The prompt to send to the model
        api_key: Gemini API key (defaults to environment variable)
    """
    client = genai.Client(
        api_key=api_key or os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-lite-preview-06-17"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")


class LLM:
    """
    A flexible LLM client that works with OpenAI-compatible APIs.
    """

    def __init__(self, url: str, model: str, api_key: str, logger=None):
        """
        Initialize the LLM client.

        Args:
            url: Base URL for the API
            model: Model name to use
            api_key: API key for authentication
            logger: Optional logger instance
        """
        self.client = OpenAI(
            base_url=url,
            api_key=api_key,
        )
        self.model = model
        self.logger = logger

    def _create_completion(self,
        messages,
        max_tokens: int = 8192,
        stream: bool = False,
        **kwargs
    ):
        """
        Create a completion using the OpenAI API.

        Args:
            messages: Messages to send to the model
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Completion object
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            **kwargs
        )
        return completion

    def _process_stream(self, stream) -> Generator[str, None, None]:
        """
        Process streaming response.

        Args:
            stream: Stream object from API

        Yields:
            Content chunks
        """
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _process_batch(self, batch) -> str:
        """
        Process batch response.

        Args:
            batch: Batch response from API

        Returns:
            Content string
        """
        return batch.choices[0].message.content.strip()

    @staticmethod
    def print_stream(stream_text: Generator[str, None, None]) -> None:
        """
        Print streaming text to console.

        Args:
            stream_text: Generator of text chunks
        """
        for chunk in stream_text:
            print(chunk, end="")

    @staticmethod
    def string2json(text: str) -> Dict[str, Any]:
        """
        Convert string to JSON object.

        Args:
            text: String containing JSON data

        Returns:
            Parsed JSON object
        """
        try:
            json_data = text.replace("```", "").replace("json", "")
            json_data = eval(json_data)
            return json_data
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(text)
            return {}

    def gen_structure_output(self, prompt: str, max_retries: int = 3,
                           default_response: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Generate structured output with retry logic.

        Args:
            prompt: Prompt to send to model
            max_retries: Maximum number of retry attempts
            default_response: Default response if all retries fail

        Returns:
            Structured response as dictionary
        """
        json_response = default_response

        for i in range(max_retries):
            try:
                completion = self._create_completion(
                    messages=prompt, stream=False
                )

                str_json = self._process_batch(batch=completion)
                json_response = self.string2json(str_json)
                break
            except Exception as e:
                if i == max_retries - 1:
                    raise e

        return json_response

    def __call__(self, messages, stream: bool = False, **kwargs) -> str:
        """
        Call the LLM with messages.

        Args:
            messages: Messages to send
            stream: Whether to stream response
            **kwargs: Additional parameters

        Returns:
            Response string or generator
        """
        completion = self._create_completion(
            messages=messages, stream=stream, **kwargs
        )

        if stream:
            return self._process_stream(stream=completion)
        else:
            return self._process_batch(batch=completion)

    @staticmethod
    def parse_tool_outputs(tool_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse tool outputs to extract successful results from content.

        Args:
            tool_outputs: List of dictionaries containing tool responses

        Returns:
            List of successful results extracted from tool outputs
        """
        successful_results = []

        for output in tool_outputs:
            try:
                content = output.get('content')
                if not content:
                    continue

                content_dict = json.loads(content)
                results = content_dict.get('successful_results', [])
                successful_results.extend(results)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON content for tool_call_id {output.get('tool_call_id')}: {e}")
            except Exception as e:
                print(f"Unexpected error for tool_call_id {output.get('tool_call_id')}: {e}")

        return successful_results


def create_litellm_client(api_key: Optional[str] = None) -> LLM:
    """
    Create a LiteLLM client instance.

    Args:
        api_key: LiteLLM API key (defaults to environment variable)

    Returns:
        Configured LLM instance
    """
    return LLM(
        url="https://litellm.distilled.ai",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key=api_key or os.environ.get("LITELLM_API_KEY")
    )


def create_localllm_client() -> LLM:
    """
    Create a local client instance.

    Args:
        api_key: API key for the provider

    Returns:
        Configured LLM instance
    """
    return LLM(
        url="http://localhost:8000/v1",
        model="Qwen/Qwen3-0.6B",
        api_key="empty"
    )

# Example usage functions
def process_crypto_query(user_question: str, web_content: str,
                        provider: str = "groq", api_key: Optional[str] = None) -> str:
    """
    Process a crypto-related query using the specified provider.

    Args:
        user_question: The user's question
        web_content: Web content to analyze
        provider: LLM provider to use ("groq", "sambanova", "gemini", "litellm")
        api_key: API key for the provider

    Returns:
        Generated response
    """
    prompt = create_crypto_summary_prompt(user_question, web_content)

    if provider == "groq":
        return generate_with_groq(prompt, api_key)
    elif provider == "sambanova":
        return generate_with_sambanova(prompt, api_key)
    elif provider == "gemini":
        # Note: Gemini prints directly, doesn't return
        generate_with_gemini(prompt, api_key)
        return ""
    elif provider == "litellm":
        llm = create_litellm_client(api_key)
        return llm(prompt)
    elif provider == "local":
        llm = create_localllm_client()
        return llm(prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Example usage:
if __name__ == "__main__":
    # Example usage
    query = "Latest news on Solana?"
    content = "Sample web content about Solana developments..."

    # Using Groq
    response = process_crypto_query(query, content, "groq")
    print("Groq Response:")
    print(response)

    # Using custom LLM client
    llm = create_litellm_client()
    response = llm("What are the latest developments in crypto?")
    print("\nLiteLLM Response:")
    print(response)
