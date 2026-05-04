import json
from datetime import datetime
from src.openrouter_client import read_message, create_message

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"transcripts/run_{timestamp}.log"

MODEL = "anthropic/claude-haiku-4.5"

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
