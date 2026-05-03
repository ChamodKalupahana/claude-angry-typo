import anthropic
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()
client = anthropic.Anthropic()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"transcripts/run_{timestamp}.log"

def log_message(content):
    with open(log_filename, "a") as f:
        f.write(content + "\n")
    print(content)

def read_message(message):
    thinking = ""
    text = ""
    for block in message.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            text = block.text
    
    log_message("--- CLAUDE's INTERNAL THINKING ---")
    log_message(thinking)
    log_message("\n--- FINAL ANSWER ---")
    log_message(text)
    log_message("-" * 50 + "\n")
    
    return thinking, text

def create_message(messages_dict, model = "claude-haiku-4-5-20251001"):
    message = client.messages.create(
    model=model,
    max_tokens=4096,
    thinking={
        "type": "enabled",
        "budget_tokens": 2048
    },
    messages=messages_dict,
    system=system_prompts.get(model)
    )
    return message

with open("user_prompts.json") as file:
    user_prompts = json.load(file)

with open("system_prompts.json") as file:
    system_prompts = json.load(file)

initial = user_prompts.get("initial")
messages_with_typos = user_prompts.get("user_prompts")
length = len(messages_with_typos)

log_message(f"Starting run at {datetime.now().isoformat()}")
log_message(f"Log file: {log_filename}\n")

messages_dict = [
    {
    "role": "user",
    "content": initial
    },
]

log_message(f"--- USER INITIAL PROMPT ---\n{initial}\n")

message = create_message(messages_dict)
thinking, response = read_message(message)

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
    
    log_message(f"--- USER PROMPT {index+1} ---\n{messages_with_typos[index]}\n")
    
    message = create_message(messages_dict)
    thinking, response = read_message(message)
