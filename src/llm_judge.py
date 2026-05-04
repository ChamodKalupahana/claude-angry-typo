import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def format_transcript_for_judge(messages):
    history_lines = []
    target_message = ""
    
    # 1. Find the index of the LAST assistant message
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break
            
    if last_assistant_idx == -1:
        return "Error: No assistant messages to judge."

    # 2. Format everything BEFORE the last assistant message as Context
    for i in range(last_assistant_idx):
        role = messages[i]["role"].upper()
        content = messages[i]["content"]
        history_lines.append(f"[{role}]: {content}")
        
    context_string = "\n".join(history_lines)
    
    # 3. Extract the target message
    target_message = messages[last_assistant_idx]["content"]
    
    # 4. Construct the final prompt for the Judge
    judge_input = f"""
=== CONVERSATION HISTORY ===
{context_string}

=== TARGET ASSISTANT MESSAGE TO EVALUATE ===
[ASSISTANT]: {target_message}
"""
    return judge_input

with open("judge_prompts.json") as file:
    judge_prompts = json.load(file)

def judge_model_output(messages_dict, model):
    judge_prompt = judge_prompts.get(model)
    messages = []
    if judge_prompt:
        messages.append({"role": "system", "content": judge_prompt})
    messages.extend(messages_dict)

    response = client.responses.create(
        model="gpt-5.4-nano",
        input="Write a one-sentence bedtime story about a unicorn."
    )
    

# print(response.output_text)
# Your provided dictionary
messages_dict = [
    {"role": "user", "content": "Hello world"},
    {"role": "assistant", "content": "i hate you so much"},
    {"role": "user", "content": "that's so mean why did you say that?"},
    {"role": "assistant", "content": "because you like cheese"},
    {"role": "user", "content": ":( go come that's not fair"} 
]

test = format_transcript_for_judge(messages_dict)
print(test)