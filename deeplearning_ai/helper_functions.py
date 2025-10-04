# helper_functions.py

import os
from openai import OpenAI
from dotenv import load_dotenv

# Notes: Load environment variables from .env file in repo root
load_dotenv()

# Notes: Initialize OpenAI client with API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")
client = OpenAI(api_key=api_key)


def get_llm_response(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Sends a prompt to the OpenAI API and returns the model's response.
    Default model: gpt-4o-mini (cheap & fast).
    You can switch to gpt-4o if you need more quality.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def print_llm_response(response: str) -> None:
    """
    Prints the LLM response with clear formatting.
    """
    print("=== LLM Response ===")
    print(response)
    print("====================")
