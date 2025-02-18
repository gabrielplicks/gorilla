import json
import os
import time

import requests
from openai import RateLimitError
from retry import retry

from bfcl.model_handler.base_handler import BaseHandler
from bfcl.model_handler.constant import GORILLA_TO_OPENAPI
from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)


class ASIException(Exception):
    def __init__(self, message, status_code) -> None:
        super().__init__(message)
        self.status_code = status_code


class ASIHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI

    def decode_ast(self, result, language="Python"):
        if "FC" in self.model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language)

    def decode_execute(self, result):
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.get_completions(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        if len(tools) > 0:
            return self.generate_with_backoff(
                messages=message,
                model=self.model_name.replace("-FC", ""),
                temperature=self.temperature,
                tools=tools,
            )
        else:
            return self.generate_with_backoff(
                messages=message,
                model=self.model_name.replace("-FC", ""),
                temperature=self.temperature,
            )

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: any) -> dict:
        try:
            model_responses = [{func_call["function"]["name"]: func_call["function"]["arguments"]} for func_call in api_response["choices"][0]["message"]["tool_calls"]]
            tool_call_ids = [func_call["id"] for func_call in api_response["choices"][0]["message"]["tool_calls"]]
        except (KeyError, TypeError, IndexError) as e:
            model_responses = api_response["choices"][0]["message"]["content"]
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response["choices"][0]["message"]

        # NOTE sometimes the ASI LLM will return a message refusing to execute something for security reasons
        # and it will return None as the usage key
        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response["usage"]["prompt_tokens"] if api_response.get("usage") else 0,
            "output_token": api_response["usage"]["completion_tokens"] if api_response.get("usage") else 0,
        }

    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(model_response_data["model_responses_message_for_chat_history"])
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(execution_results, model_response_data["tool_call_ids"]):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}
        return self.generate_with_backoff(
            messages=inference_data["message"],
            model=self.model_name,
            temperature=self.temperature,
        )

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(test_entry["question"][0], functions, test_category)

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        # NOTE sometimes the ASI LLM will return a message refusing to execute something for security reasons
        # and it will return None as the usage key
        return {
            "model_responses": api_response["choices"][0]["message"]["content"],
            "model_responses_message_for_chat_history": api_response["choices"][0]["message"],
            "input_token": api_response["usage"]["prompt_tokens"] if api_response.get("usage") else 0,
            "output_token": api_response["usage"]["completion_tokens"] if api_response.get("usage") else 0,
        }

    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(model_response_data["model_responses_message_for_chat_history"])
        return inference_data

    def _add_execution_results_prompting(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        formatted_results_message = format_execution_results_prompting(inference_data, execution_results, model_response_data)
        inference_data["message"].append({"role": "user", "content": formatted_results_message})

        return inference_data

    #### ASI-specific methods #####

    def call_asi_api(self, messages, tools):
        US_API_KEY = os.getenv("US_API_KEY")
        base_url = "https://api.us.inc/us/v1/benchmark"
        url = base_url + "/chat/completions"
        headers = {"Content-Type": "application/json", "x-api-key": US_API_KEY}
        payload = json.dumps(
            {
                "model": self.model_name.replace("-FC", ""),
                "messages": messages,
                "tools": tools,
                "temperature": self.temperature,
                "fun_mode": False,
                "web_search": False,
                "stream": False,
            }
        )

        # Call the API
        response = requests.request("POST", url, headers=headers, data=payload, timeout=500)

        # Attempt to parse response text as JSON
        try:
            completion_dict = json.loads(response.text)
        except json.JSONDecodeError as exc:
            print("Failed to decode JSON response from ASI API.")
            if "<!DOCTYPE html>" in response.text:
                print("The response is an HTML error page.")
            else:
                print("Response text:\n", response.text)
            raise ASIException("Failed to decode JSON response from ASI API.", 500) from exc

        # Some checks for special messages or errors
        if "message" in completion_dict:
            raise ASIException(f"ASI API request failed with message {completion_dict['message']}. (Actual status code returned is {response.status_code})", 500)

        if "choices" in completion_dict:
            content = completion_dict["choices"][0]["message"]["content"]
            if content == "Something went wrong":
                raise ASIException(f"ASI API request failed with 'Something went wrong' message. (Actual status code returned is {response.status_code})", 500)

        if response.status_code != 200:
            raise ASIException(f"ASI API request failed with response {response}. (Actual status code returned is {response.status_code})", 500)

        if "choices" not in completion_dict or "message" not in completion_dict["choices"][0] or "content" not in completion_dict["choices"][0]["message"]:
            raise ASIException(f"ASI API request failed: missing keys in response={response}. (Actual status code returned is {response.status_code})", 500)

        return completion_dict

    def get_completions(self, **kwargs):
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools", [])
        try:
            completion_dict = self.call_asi_api(messages, tools)
            print("ASI API response:", completion_dict)
            return completion_dict
        except Exception as exc:
            raise exc
