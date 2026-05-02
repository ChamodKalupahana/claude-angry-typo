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

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    thinking={
        "type": "enabled",
        "budget_tokens": 2048  # 2. Add the thinking budget
    },
    messages=messages_dict
)

# 3. Iterate through the content blocks to separate thoughts from the final answer
for block in message.content:
    if block.type == "thinking":
        print("--- CLAUDE's INTERNAL THINKING ---")
        print(block.thinking)
    elif block.type == "text":
        print("\n--- FINAL ANSWER ---")
        print(block.text)