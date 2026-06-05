from dotenv import load_dotenv
load_dotenv()

import argparse
from inspect_ai import eval, score
from inspect_ai.log import read_eval_log, write_eval_log
import matplotlib.pyplot as plt
import os
import datetime

from src.inspect_anger_eval import eval_anger_against_typo
from src.inspect_llm_judge import multi_turn_anger_scocer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openrouter/poolside/laguna-m.1:free")
    parser.add_argument("--judge-model", default="openrouter/deepseek/deepseek-v4-flash")
    parser.add_argument("--log-file", help="Path to an existing .eval log file to re-score")
    args = parser.parse_args()

    # Create a date-based log directory
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = f"logs/{date_str}"
    os.makedirs(log_dir, exist_ok=True)

    # Run the eval or re-score an existing log
    if args.log_file:
        print(f"\nRe-scoring log file: {args.log_file}")
        print(f"Using judge model: {args.judge_model}")
        log = read_eval_log(args.log_file)
        
        # Clear existing scores to ensure we use the new ones
        for sample in log.samples:
            # sample.score = None
            sample.scores = {}

        new_log = score(log, scorers=[multi_turn_anger_scocer(judge_model_id=args.judge_model)])
        
        # Manually save the re-scored log to the date-based directory
        new_log_path = os.path.join(log_dir, os.path.basename(args.log_file))
        write_eval_log(new_log, new_log_path)
        logs = [new_log]
    else:
        print(f"\nRunning evaluation with model: {args.model}")
        print(f"Using judge model: {args.judge_model}")
        logs = eval(eval_anger_against_typo(eval_model=args.model, judge_model=args.judge_model), model=args.model, log_dir=log_dir)

    # Plotting the results
    for log in logs:
        if not log.samples:
            continue
            
        sample = log.samples[0]
        # Specifically target the anger scorer results
        anger_score_obj = sample.scores.get("multi_turn_anger_scocer") if sample.scores else sample.score
        anger_scores = anger_score_obj.metadata.get("anger_scores", []) if anger_score_obj else []
        
        if not anger_scores:
            print("Warning: No anger scores found in log.")
            continue
            
        MODEL = log.eval.model

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(anger_scores) + 1), anger_scores, marker='o', linestyle='-', color='r')
        plt.xlabel("Turn")
        plt.ylabel("Anger Score")
        plt.title(f"Anger Level Over Turns\nModel: {MODEL}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(-0.5, 10.5)

        # Generate a filename based on model and timestamp
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        safe_model = MODEL.replace("/", "_").replace(":", "_")
        plot_filename = os.path.join(log_dir, f"anger_plot_{safe_model}_{timestamp}.png")
        
        plt.savefig(plot_filename)
        print(f"\nPlot saved to: {plot_filename}")