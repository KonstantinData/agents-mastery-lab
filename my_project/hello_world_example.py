import os
from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv()  # Load .env and set environment variables

# If needed explicitly:
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")

agent = Agent(name="Assistant", instructions="You are a helpful assistant")
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
