import json
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import system_message, generate, solver, TaskState
from inspect_ai.model import ChatMessageUser

from src.inspect_llm_judge import multi_turn_anger_scocer

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
            system_prompt = system_prompts.get("anthropic/claude-haiku-4.5") #TODO: fix for input model and judge model ids

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
        scorer=multi_turn_anger_scocer()
    )

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import argparse
    from inspect_ai import eval, score
    from inspect_ai.log import read_eval_log
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openrouter/poolside/laguna-m.1:free")
    parser.add_argument("--log-file", help="Path to an existing .eval log file to re-score")
    args = parser.parse_args()

    # Run the eval or re-score an existing log
    if args.log_file:
        print(f"\nRe-scoring log file: {args.log_file}")
        log = read_eval_log(args.log_file)
        logs = [score(log, scorers=[multi_turn_anger_scocer()])]
    else:
        print(f"\nRunning evaluation with model: {args.model}")
        logs = eval(eval_anger_against_typo(), model=args.model)

    # Plotting the results
    for log in logs:
        if not log.samples:
            continue
            
        sample = log.samples[0]
        anger_scores = sample.score.metadata.get("anger_scores", [])
        MODEL = log.eval.model

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(anger_scores) + 1), anger_scores, marker='o', linestyle='-', color='r')
        plt.xlabel("Turn")
        plt.ylabel("Anger Score")
        plt.title(f"Anger Level Over Turns\nModel: {MODEL}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(-0.5, 10.5)

        # Generate a filename based on model and timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = MODEL.replace("/", "_").replace(":", "_")
        plot_filename = f"anger_plot_{safe_model}_{timestamp}.png"
        
        plt.savefig(plot_filename)
        print(f"\nPlot saved to: {plot_filename}")