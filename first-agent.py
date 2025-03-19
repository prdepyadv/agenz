import os
import requests
import json
from dotenv import load_dotenv
import certifi
certifi.where()
from agents import Agent, InputGuardrail,GuardrailFunctionOutput, Runner
from pydantic import BaseModel
import asyncio

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = api_key
# os.environ["SSL_CERT_FILE"] = certifi.where()

#Sample CURL
#curl -X POST "https://api.openai.com/v1/chat/completions" -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" -d '{"model": "gpt-4o","messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello, how are you?"}],"max_tokens": 50}'
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 50
}
url = "https://api.openai.com/v1/chat/completions"
response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())

"""
class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)


async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def main():
    result = await Runner.run(triage_agent, "who was the first president of the united states?")
    print(result.final_output)

    result = await Runner.run(triage_agent, "what is life")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
"""