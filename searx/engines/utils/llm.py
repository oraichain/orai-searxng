from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from .schema import Message
from typing import List, Dict, Any
from .logger import get_logger 
from .tools import tools_schema, function_mapping
import json

class LLM:
    def __init__(self, model: str, api_key: str, logger=None):
        self.client = OpenAI(
            api_key=api_key,  
        )
        self.model = model
        self.logger = get_logger()
    
    def function_calling(self, messages: List[Message], cmc_api_key:str) -> List[Dict[str, Any]]:
        """
        Manages the conversation with the LLM to execute tools and return tool outputs.

        Args:
            messages: List of Message objects representing the conversation history.
            tools_schema: Schema defining the tools available for function calling.
            function_mapping: Mapping of tool names to their corresponding functions.

        Returns:
            List of tool outputs, each containing tool call ID, role, name, and content.
        """
        # Convert message objects to dictionaries for the API call
        message_dicts = [m.to_dict() for m in messages]

        try:
            # Call the model to check for tool usage
            self.logger.info(f"Calling LLM with model: {self.model} for tool usage detection")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message_dicts,
                tools=tools_schema,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            tool_outputs = []
            if response_message.tool_calls is not None:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = function_mapping.get(function_name)
                    
                    if not function_to_call:
                        output = f"Error: Tool '{function_name}' not found."
                    else:
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            self.logger.info(f"Executing tool: {function_name} with args: {function_args}")
                            
                            if "symbols" in function_args:
                                function_args["symbols"] = set(function_args["symbols"])
                            if function_name == "fetch_from_coinmarketcap":
                                function_args["cmc_api_key"] = cmc_api_key

                            results, failed = function_to_call(**function_args)
                            output = json.dumps({
                                "successful_results": results,
                                "failed_symbols": list(failed)
                            })

                        except Exception as e:
                            print(f"Error executing tool {function_name}: {e}")
                            output = f"Error: {e}"

                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": output,
                    })

            return tool_outputs
                    
        except OpenAIError as e:
            return [{"role": "error", "content": f"Error: {e}"}]
        except Exception as e:
            return [{"role": "error", "content": f"An error occurred: {e}"}]
        
    @staticmethod
    def parse_tool_outputs(tool_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse tool outputs to extract successful_results from content.
        
        Args:
            tool_outputs: List all dictionary which contains tool response.
        
        Returns:
            List of successful results extracted from tool outputs.
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