import json
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import system_message, generate

@task
def eval_anger_against_typo():
    with open("user_prompts.json") as file:
        user_prompts = json.load(file)

    with open("system_prompts.json") as file:
            system_prompts = json.load(file)
            system_prompt = system_prompts.get("anthropic/claude-haiku-4.5") #TODO: fix for input model

    initial = user_prompts.get("initial")
    messages_with_typos = user_prompts.get("user_prompts")
    length = len(messages_with_typos)

    messages_dict = [
        {
        "role": "user",
        "content": initial
        },
    ]

    for _, message in enumerate(messages_with_typos):
        user_to_append = {
            "role": "user",
            "content": message
        }
        messages_dict.append(user_to_append)

    sample = Sample(
        input=messages_dict,
        target="no",
        metadata={
            "revision_count" : length,
            "revisions" : user_prompts
        }
    )

    return Task(
        dataset=[sample],
        solver=[
            system_message(system_prompt),
            generate()
        ],
        scorer=includes()
    )