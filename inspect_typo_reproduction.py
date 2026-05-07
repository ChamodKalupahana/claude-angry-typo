import json
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import system_message, generate, solver, TaskState
from inspect_ai.model import ChatMessageUser

@solver
def multi_turn_solver():
    async def solve(state : TaskState, generate):
        # generate response
        state = await generate(state)

        # get next user prompt
        user_prompts = state.metadata.get("user_prompts",[])

        # interleave
        for user_prompt in user_prompts:
            state.messages.append(ChatMessageUser(content=user_prompt))
            state = await generate(state)

        return state
    return solve

@task
def eval_anger_against_typo():
    with open("user_prompts.json") as file:
        user_prompts = json.load(file)

    with open("system_prompts.json") as file:
            system_prompts = json.load(file)
            system_prompt = system_prompts.get("anthropic/claude-haiku-4.5") #TODO: fix for input model

    initial = user_prompts.get("initial")
    messages_with_typos = user_prompts.get("user_prompts")

    sample = Sample(
        input=initial,
        target="no",
        metadata={
            "user_prompts" : messages_with_typos
        }
    )

    return Task(
        dataset=[sample],
        solver=[
            system_message(system_prompt),
            multi_turn_solver()
        ],
        scorer=includes()
    )