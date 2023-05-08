import argparse
import os

import llm_feedback.pilot.tasks as tasks
from llm_feedback.utils.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_outputs_path", type=str)  # path to JSONL
    parser.add_argument("--task", type=str)
    parser.add_argument("--task_args_str", type=str, default=None)
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--override_filename", type=str, default=None)
    # if override_filename is not provided, we replace "__outputs.jsonl" with "__metrics.json"
    args = parser.parse_args()

    task = tasks.get_task(task_name=args.task, task_args_str=args.task_args_str)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.override_filename is None:
        filename = args.model_outputs_path.replace("__outputs.jsonl", "__metrics.json")
    else:
        filename = args.override_filename
    write_path = os.path.join(args.output_dir, filename)

    model_outputs = read_jsonl(args.model_outputs_path)
    metrics = task.evaluate(phase=args.phase, outputs=model_outputs)
    write_json(metrics, write_path)

    print(f"Wrote {args.task} metrics to {write_path}.")


if __name__ == "__main__":
    main()
