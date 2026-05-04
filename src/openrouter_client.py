import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def read_message(message, log_callback=None):
    thinking = ""
    text = ""
    
    text = message.choices[0].message.content
    
    # Try to get reasoning/thinking content
    if hasattr(message.choices[0].message, 'reasoning_content'):
        thinking = message.choices[0].message.reasoning_content
    elif message.choices[0].message.model_extra and 'reasoning' in message.choices[0].message.model_extra:
         thinking = message.choices[0].message.model_extra['reasoning']
    
    if log_callback:
        log_callback("--- CLAUDE's INTERNAL THINKING ---")
        log_callback(thinking if thinking else "[No thinking content returned]")
        log_callback("\n--- FINAL ANSWER ---")
        log_callback(text)
        log_callback("-" * 50 + "\n")
    
    return thinking, text

def create_message(messages_dict, system_prompts, model="anthropic/claude-4.6-sonnet"):
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
