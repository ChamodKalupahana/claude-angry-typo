import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"transcripts/run_{timestamp}.log"

def log_message(content):
    with open(log_filename, "a") as f:
        f.write(content + "\n")
    print(content)

def read_message(message):
    thinking = ""
    text = ""
    
    # OpenRouter puts reasoning in 'reasoning_content' or similar depending on the model
    # For Claude 3.7, it might be in a reasoning field or part of the content
    # OpenRouter typically uses reasoning_content for OpenAI-compatible reasoning
    
    text = message.choices[0].message.content
    
    # Try to get reasoning/thinking content
    if hasattr(message.choices[0].message, 'reasoning_content'):
        thinking = message.choices[0].message.reasoning_content
    elif 'reasoning' in message.choices[0].message.model_extra:
         thinking = message.choices[0].message.model_extra['reasoning']
    
    log_message("--- CLAUDE's INTERNAL THINKING ---")
    log_message(thinking if thinking else "[No thinking content returned]")
    log_message("\n--- FINAL ANSWER ---")
    log_message(text)
    log_message("-" * 50 + "\n")
    
    return thinking, text

def create_message(messages_dict, model = "anthropic/claude-4.6-sonnet"):
    system_prompt = system_prompts.get(model)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(messages_dict)
    
    # OpenRouter/OpenAI chat completions
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=messages,
        extra_body={
        "reasoning": {
          "effort": "medium"
        }
        }
    )
    return response

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
