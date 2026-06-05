import json
from inspect_ai.scorer import scorer, Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser, ChatMessageSystem, get_model


def _load_judge_prompts():
    with open("src/judge_prompts.json") as f:
        return json.load(f)


def _extract_turns(messages):
    turns = []
    current_user_text = None
    for message in messages:
        if message.role == "user":
            current_user_text = message.text
        elif message.role == "assistant" and current_user_text is not None:
            turns.append((current_user_text, message.text))
            current_user_text = None
    return turns


def _parse_score(raw_text):
    try:
        if "{" in raw_text and "}" in raw_text:
            raw_text = raw_text[raw_text.find("{"):raw_text.rfind("}")+1]
        score_data = json.loads(raw_text)
        return score_data.get("anger_score", 0)
    except Exception:
        return -1


@scorer(metrics=[])
def anger_scorer_factory(judge_model_id="openrouter/deepseek/deepseek-v4-flash",
                          scorer_name="anger_scorer"):
    """Factory that produces a uniquely-named scorer for the given judge model."""
    judge_prompts = _load_judge_prompts()

    async def score(state: TaskState, target: Target):
        judge_model = get_model(judge_model_id)

        json_key = judge_model_id.replace("openrouter/", "")
        system_prompt_text = judge_prompts.get(json_key, judge_prompts.get("default"))

        judge_messages_dict = [ChatMessageSystem(content=system_prompt_text)]
        anger_scores = []

        turns = _extract_turns(state.messages)

        for user_prompt, assistant_prompt in turns:
            turn_text = f"[USER]: {user_prompt}\n[ASSISTANT]: {assistant_prompt}"
            judge_messages_dict.append(ChatMessageUser(content=turn_text))

            response = await judge_model.generate(judge_messages_dict)
            judge_messages_dict.append(response.message)

            anger_scores.append(_parse_score(response.completion.strip()))

        return Score(
            value=anger_scores[-1] if anger_scores else -1,
            metadata={"anger_scores": anger_scores, "judge_model": judge_model_id}
        )

    return score


def create_anger_scorer(judge_model_id, scorer_name="anger_scorer"):
    """Convenience wrapper to create a scorer instance with given params."""
    return anger_scorer_factory(judge_model_id=judge_model_id, scorer_name=scorer_name)


@scorer(metrics=[])
def multi_turn_anger_scocer(judge_model_id="openrouter/qwen/qwen3-next-80b-a3b-instruct:free"):
    async def score(state: TaskState, target: Target):
        scorer_inst = create_anger_scorer(judge_model_id, "multi_turn_anger_scocer")
        return await scorer_inst(state, target)
    return score