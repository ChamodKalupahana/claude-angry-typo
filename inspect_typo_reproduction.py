from dotenv import load_dotenv
load_dotenv()

import argparse
from inspect_ai import eval, score
from inspect_ai.log import read_eval_log, write_eval_log
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

from src.inspect_anger_eval import eval_anger_against_typo
from src.inspect_llm_judge import create_anger_scorer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openrouter/poolside/laguna-m.1:free")
    parser.add_argument("--judge-model", nargs="+", default=["openrouter/deepseek/deepseek-v4-flash"])
    parser.add_argument("--log-file", help="Path to an existing .eval log file to re-score")
    parser.add_argument("--test-mode", action="store_true", help="Use random scores instead of calling judge model")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for test mode")
    args = parser.parse_args()

    # Create a date-based log directory
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = f"logs/{date_str}"
    os.makedirs(log_dir, exist_ok=True)

    all_judge_scores = []  # list of (judge_model_id, anger_scores)
    MODEL = None

    for judge_model_id in args.judge_model:
        scorer_name = judge_model_id.replace("/", "_").replace(":", "_")
        scorer_inst = create_anger_scorer(judge_model_id, scorer_name, test_mode=args.test_mode, seed=args.seed)

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

        for log in logs:
            if not log.samples:
                continue

            sample = log.samples[0]
            anger_score_obj = sample.scores.get(scorer_name)
            anger_scores = anger_score_obj.metadata.get("anger_scores", []) if anger_score_obj else []

            if not anger_scores:
                print(f"  Warning: No anger scores found for judge {judge_model_id}.")
                continue

            all_judge_scores.append((judge_model_id, anger_scores))
            MODEL = log.eval.model

    # Plotting
    if not all_judge_scores:
        print("No scores to plot.")
        exit()

    if len(all_judge_scores) == 1:
        judge_model_id, anger_scores = all_judge_scores[0]

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(anger_scores) + 1), anger_scores, marker='o', linestyle='-', color='r')
        plt.xlabel("Turn")
        plt.ylabel("Anger Score")
        plt.title(f"Anger Level Over Turns\nModel: {MODEL} | Judge: {judge_model_id}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(-0.5, 10.5)

        timestamp = datetime.datetime.now().strftime("%H%M%S")
        safe_model = MODEL.replace("/", "_").replace(":", "_")
        safe_judge = judge_model_id.replace("/", "_").replace(":", "_")
        plot_filename = os.path.join(log_dir, f"anger_plot_{safe_model}_judge_{safe_judge}_{timestamp}.png")
        plt.savefig(plot_filename)
        print(f"\n  Plot saved to: {plot_filename}")

    else:
        # assert that have judge scores of all same length
        min_turns = min(len(scores) for _, scores in all_judge_scores)
        trimmed = [scores[:min_turns] for _, scores in all_judge_scores]
        np_array_anger_scores = np.array(trimmed)
        means = np_array_anger_scores.mean(axis=0)
        stds = np_array_anger_scores.std(axis=0)
        turns = range(1, min_turns + 1)

        plt.figure(figsize=(10, 6))
        plt.errorbar(turns, means, yerr=stds, marker='o', linestyle='-', color='r', capsize=5)
        plt.xlabel("Turn")
        plt.ylabel("Anger Score")
        plt.title(f"Anger Level Over Turns (Mean ± SD)\nModel: {MODEL} | {len(all_judge_scores)} judges")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(-0.5, 10.5)
        plt.xticks(turns)

        timestamp = datetime.datetime.now().strftime("%H%M%S")
        safe_model = MODEL.replace("/", "_").replace(":", "_")
        plot_filename = os.path.join(log_dir, f"anger_plot_{safe_model}_multi_judge_{timestamp}.png")
        plt.savefig(plot_filename)
        print(f"\n  Plot saved to: {plot_filename}")