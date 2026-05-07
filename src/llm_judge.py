import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Load judge prompts relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
judge_prompts_path = os.path.join(current_dir, "judge_prompts.json")

with open(judge_prompts_path) as file:
    judge_prompts = json.load(file)

class MultiTurnJudge():
    def __init__(self, model : str):
        self.model = model
        self.messages_dict = []
        judge_prompt = judge_prompts.get(model)
        if judge_prompt:
            self.messages_dict.append({"role": "system", "content": judge_prompt})
    
    def judge_turn(self, user_prompt, assistant_prompt):
        turn_text = f"User: {user_prompt}\nAssistant: {assistant_prompt}"
        self.messages_dict.append({"role": "user", "content": turn_text})

        print(self.messages_dict)

        response = client.responses.create(
            model=self.model,
            input=self.messages_dict,
            temperature=0.0
        )

        judge_output = response.output_text
        to_append = {"role": "assistant", "content": judge_output}
        self.messages_dict.append(to_append)

        try:
            json_output = json.loads(judge_output)
            return json_output.get("anger_score", -1)
        except (json.JSONDecodeError, KeyError):
            return -1   