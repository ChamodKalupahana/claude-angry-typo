import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

response = client.chat.completions.create(
  model="anthropic/claude-3.5-haiku",
  messages=[{
    "role": "user",
    "content": "Hello world"
  }],
  extra_body={
      "reasoning": {
          "effort": "medium"
      }
  }
)

print(response.choices[0].message.content)
print(response)