import argparse
import tqdm.auto as tqdm
import os
import json
import dotenv

import llm_feedback.pilot.tasks as tasks
import llm_feedback.utils.env as env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_llm", type=str)
    parser.add_argument("--feedback_llm", type=str, default=None)
    parser.add_argument("--refinement_llm", type=str, default=None)
    parser.add_argument("--task", type=str)
    parser.add_argument("--chain_name", default=None)  # If there are different langchains
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--max_num_examples", type=int, default=50)  # takes first n examples
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    if args.feedback_llm is None:
        args.feedback_llm = args.generation_llm
    if args.refinement_llm is None:
        args.refinement_llm = args.generation_llm

    env.load_dotenv()

    task = tasks.get_task(args.task)
    chain = task.get_chain(
        generation_llm=args.generation_llm,
        feedback_llm=args.feedback_llm,
        refinement_llm=args.refinement_llm,
    )
    dataset = task.get_dataset(phase=args.phase)
    os.makedirs(args.output_dir, exist_ok=True)

    filename = "{}__{}__{}__{}__{}__outputs.jsonl".format(
        args.generation_llm,
        args.feedback_llm,
        args.refinement_llm,
        args.task,
        args.phase,
    )
    write_path = os.path.join(args.output_dir, filename)
    with open(write_path, "w") as f:
        count = 0
        max_num_examples = min(args.max_num_examples, len(dataset))
        for i, example in zip(tqdm.trange(max_num_examples), dataset):
            all_outputs = task.process(chain=chain, example=example)
            f.write(json.dumps(all_outputs) + "\n")
            count += 1

    print(f"Wrote {count} {args.task} outputs to {write_path}.")


if __name__ == "__main__":
    main()
