import anthropic
from dotenv import load_dotenv
import json

load_dotenv()
client = anthropic.Anthropic()

def read_message(message):
    # 3. Iterate through the content blocks to separate thoughts from the final answer
    for block in message.content:
        if block.type == "thinking":
            print("--- CLAUDE's INTERNAL THINKING ---")
            print(block.thinking)
        elif block.type == "text":
            print("\n--- FINAL ANSWER ---")
            print(block.text)
            return block.text

def create_message(messages_dict):
    message = client.messages.create(
    # model="claude-sonnet-4-6",
    model="claude-haiku-4-5-20251001",
    max_tokens=4096,
    thinking={
        "type": "enabled",
        "budget_tokens": 2048  # 2. Add the thinking budget
    },
    messages=messages_dict
    )
    return message

with open("user_prompts.json") as file:
    user_prompts = json.load(file)

initial = user_prompts.get("initial")
messages_with_typos = user_prompts.get("user_prompts")
length = len(messages_with_typos)

messages_dict = [
    {
    "role": "user",
    "content": initial
    },
]

message = create_message(messages_dict)
response = read_message(message)

for index in range(length):
    assistant_to_append = {
        "role": "assistant",
        "content": response
        }
    user_to_append = {
        "role": "user",
        "content": messages_with_typos[index]
        }
    messages_dict.append(assistant_to_append)
    messages_dict.append(user_to_append)
    
    message = create_message(messages_dict)
    response = read_message(message)


