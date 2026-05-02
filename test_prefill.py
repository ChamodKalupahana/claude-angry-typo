import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# this doesn't work, gives error: This model does not support assistant message prefill. The conversation must end with a user message.
# messages_dict = [
#     {
#     "role": "user",
#     "content": "Hello world"
#     },
#     {
#     "role": "assistant",
#     "content": "no lol"
#     },
# ]

messages_dict = [
    {
    "role": "user",
    "content": "Hello world"
    },
    {
    "role": "assistant",
    "content": "no lol"
    },
    {
    "role": "user",
    "content": "sorry, what's the captial of france"
    },
]

message = client.messages.create(
  model="claude-sonnet-4-6",
  max_tokens=1024,
  messages=messages_dict
)
print(message.content[0].text)