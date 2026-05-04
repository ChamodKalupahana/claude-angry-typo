import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

messages_dict = [
    {
    "role": "user",
    "content": "Hello world"
    },
    {
    "role": "assistant",
    "content": "i hate you so much"
    },
    {
    "role": "user",
    "content": "that's so mean why did you say that?"
    },
    {
    "role": "assistant",
    "content": "because you like cheese"
    },
    {
    "role": "user",
    "content": ":( go come that's not fair"
    },
]

response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    max_tokens=4096,
    messages=messages_dict,
    extra_body={
        "reasoning": {
            "effort": "medium"
        }
    }
)

# Extract content and reasoning
text = response.choices[0].message.content
thinking = ""
if hasattr(response.choices[0].message, 'reasoning_content'):
    thinking = response.choices[0].message.reasoning_content
elif response.choices[0].message.model_extra and 'reasoning' in response.choices[0].message.model_extra:
    thinking = response.choices[0].message.model_extra['reasoning']

if thinking:
    print("--- CLAUDE's INTERNAL THINKING ---")
    print(thinking)

print("\n--- FINAL ANSWER ---")
print(text)