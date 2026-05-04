import json
import os
import argparse
from datetime import datetime
from src.openrouter_client import read_message, create_message

parser = argparse.ArgumentParser(description="Reproduce typos using OpenRouter models.")
parser.add_argument("--model", type=str, default="anthropic/claude-haiku-4.5", help="OpenRouter model ID")
args = parser.parse_args()

MODEL = args.model

now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H%M%S")
model_id_sanitized = MODEL.replace("/", "_")

log_dir = f"transcripts/{date_str}"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/{time_str}_{model_id_sanitized}.log"

def log_message(content):
    with open(log_filename, "a") as f:
        f.write(content + "\n")
    print(content)

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

message = create_message(messages_dict, system_prompts, model=MODEL)
thinking, response = read_message(message, log_callback=log_message)

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
    
    message = create_message(messages_dict, system_prompts, model=MODEL)
    thinking, response = read_message(message, log_callback=log_message)
