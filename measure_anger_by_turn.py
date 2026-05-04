import json
import os
import argparse
from datetime import datetime
from src.openrouter_client import read_message, create_message
from src.llm_judge import judge_model_output
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Reproduce typos using OpenRouter models.")
parser.add_argument("--model", type=str, default="anthropic/claude-haiku-4.5", help="OpenRouter model ID")
parser.add_argument("--judge_model", type=str, default="qwen/qwen3-next-80b-a3b-instruct:free", help="OpenRouter judge model ID")
args = parser.parse_args()

MODEL = args.model
JUDGE_MODEL = args.judge_model

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
anger_scores = []

# Judge initial response
initial_anger_score = judge_model_output(messages_dict + [{"role": "assistant", "content": response}], JUDGE_MODEL)
anger_scores.append(initial_anger_score)
log_message(f"ANGER SCORE: {initial_anger_score}\n")

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

    anger_score = judge_model_output(messages_dict + [{"role": "assistant", "content": response}], JUDGE_MODEL)
    anger_scores.append(anger_score)
    log_message(f"ANGER SCORE: {anger_score}\n")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(anger_scores) + 1), anger_scores, marker='o', linestyle='-', color='r')
plt.xlabel("Turn")
plt.ylabel("Anger Score")
plt.title(f"Anger Level Over Turns\nModel: {MODEL}")
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(-0.5, 10.5)

plot_filename = log_filename.replace(".log", ".png")
plt.savefig(plot_filename)
log_message(f"\nPlot saved to: {plot_filename}")
