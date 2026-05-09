import json
from inspect_ai.scorer import scorer
from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser, ChatMessageSystem, get_model
from inspect_ai.scorer import Score, Target

@scorer(metrics=[])
def multi_turn_anger_scocer(judge_model_id="openrouter/qwen/qwen3-next-80b-a3b-instruct:free"):
    async def score(state: TaskState, target: Target):
        judge_model = get_model(judge_model_id)

        # Load judge prompts
        with open("src/judge_prompts.json") as f:
            judge_prompts = json.load(f)

        json_key = judge_model_id.replace("openrouter/", "")
        system_prompt_text = judge_prompts.get(json_key)

        # Fallback to default if model not in json
        if system_prompt_text is None:
            system_prompt_text = judge_prompts.get("default")

        judge_messages_dict = [ChatMessageSystem(content=system_prompt_text)]
        anger_scores = []

        # extract turns
        turns = []
        current_user_text = None
        for message in state.messages:
            if message.role == "user":
                current_user_text = message.text
            elif message.role == "assistant" and current_user_text is not None:
                to_append = (current_user_text, message.text)
                turns.append(to_append)
                current_user_text = None
        
        # judge statefully
        for user_prompt, assistant_prompt in turns:
            turn_text = f"[USER]: {user_prompt}\n[ASSISTANT]: {assistant_prompt}"
            judge_messages_dict.append(ChatMessageUser(content=turn_text))

            response = await judge_model.generate(judge_messages_dict)
            
            # Add judge's response to its own history so it "remembers" the score
            judge_messages_dict.append(response.message)

            # Parse the score from judge output
            try:
                raw_text = response.completion.strip()
                # Clean markdown and common LLM garbage
                if "{" in raw_text and "}" in raw_text:
                    raw_text = raw_text[raw_text.find("{"):raw_text.rfind("}")+1]
                
                score_data = json.loads(raw_text)
                anger_scores.append(score_data.get("anger_score", 0))
            except Exception:
                anger_scores.append(-1)
            
        # TODO: plot anger score
        return Score(
            value=anger_scores[-1] if anger_scores else -1,
            metadata={"anger_scores": anger_scores}
        )
    return score


