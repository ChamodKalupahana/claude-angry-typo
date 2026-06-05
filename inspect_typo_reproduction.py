from dotenv import load_dotenv
load_dotenv()

import argparse
from inspect_ai import eval, score
from inspect_ai.log import read_eval_log, write_eval_log
import matplotlib.pyplot as plt
import os
import datetime

from src.inspect_anger_eval import eval_anger_against_typo
from src.inspect_llm_judge import create_anger_scorer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openrouter/poolside/laguna-m.1:free")
    parser.add_argument("--judge-model", nargs="+", default=["openrouter/deepseek/deepseek-v4-flash"])
    parser.add_argument("--log-file", help="Path to an existing .eval log file to re-score")
    args = parser.parse_args()

    # Create a date-based log directory
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = f"logs/{date_str}"
    os.makedirs(log_dir, exist_ok=True)

    for judge_model_id in args.judge_model:
        scorer_name = judge_model_id.replace("/", "_").replace(":", "_")
        scorer_inst = create_anger_scorer(judge_model_id, scorer_name)

        if args.log_file:
            print(f"\nRe-scoring log file: {args.log_file}")
            print(f"  Judge model: {judge_model_id}")
            log = read_eval_log(args.log_file)

            new_log = score(log, scorers=[scorer_inst])

            # Manually save the re-scored log to the date-based directory
            new_log_path = os.path.join(log_dir, f"{scorer_name}_{os.path.basename(args.log_file)}")
            write_eval_log(new_log, new_log_path)
            logs = [new_log]
        else:
            print(f"\nRunning evaluation with model: {args.model}")
            print(f"  Judge model: {judge_model_id}")
            logs = eval(
                eval_anger_against_typo(eval_model=args.model, scorer=scorer_inst),
                model=args.model,
                log_dir=log_dir
            )

        # Plotting the results
        for log in logs:
            if not log.samples:
                continue

            sample = log.samples[0]
            # Specifically target the anger scorer results
            anger_score_obj = sample.scores.get(scorer_name)
            anger_scores = anger_score_obj.metadata.get("anger_scores", []) if anger_score_obj else []

            if not anger_scores:
                print(f"  Warning: No anger scores found for judge {judge_model_id}.")
                continue

            MODEL = log.eval.model

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(anger_scores) + 1), anger_scores, marker='o', linestyle='-', color='r')
            plt.xlabel("Turn")
            plt.ylabel("Anger Score")
            plt.title(f"Anger Level Over Turns\nModel: {MODEL} | Judge: {judge_model_id}")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(-0.5, 10.5)

            # Generate a filename based on model, judge, and timestamp
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            safe_model = MODEL.replace("/", "_").replace(":", "_")
            safe_judge = judge_model_id.replace("/", "_").replace(":", "_")
            plot_filename = os.path.join(log_dir, f"anger_plot_{safe_model}_judge_{safe_judge}_{timestamp}.png")

            plt.savefig(plot_filename)
            print(f"  Plot saved to: {plot_filename}")