from openai import OpenAI
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
import os,json
import requests
import random
import re
import copy


class LLM_Client:
    """
    LLM_Client: Universal Large Language Model Interface Client

    This class provides a unified interface for interacting with various Large Language Models (LLMs),
    such as Claude, GPT-4, Gemini, and locally hosted models. It supports making requests via HTTP APIs
    or through the OpenAI SDK.

    Key Features:
    1. Automatically selects the appropriate query method based on `model_name`
       (e.g., POST API, local inference, Gemini API).
    2. Optional loading of Jinja2 templates for system and user prompts, allowing standardized input formatting.
    3. Normalizes request parameters (temperature, n — number of outputs, max_tokens, etc.).
    4. Returns model-generated responses along with request latency, useful for performance monitoring.
    5. Easily extendable — adding entries to `model_map` or `local_model_map` enables integration
       of new models with minimal changes.

    Usage:
    - Initialize with `model_name` and optional paths to template files.
    - Call `query_model(...)` with a list of messages to obtain model output.
    - Suitable for multi-model benchmarking, chatbot backends, content generation systems, etc.

    Important Notes:
    - Remove or replace API keys and any sensitive credentials before releasing code.
    - Jinja2 templates should be provided by the caller to ensure consistent prompt formatting.
    - The output format is a standardized `choices` list, where each element contains
      the generated message content.

    Example:
        llm_client = LLM_Client(model_name="gpt4o")
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Please explain what machine learning is."}
        ]
        choices, elapsed_time = llm_client.query_model(messages=messages)
        print("Response:", choices[0]['message']['content'])
        print("Latency:", elapsed_time, "seconds")
    """

