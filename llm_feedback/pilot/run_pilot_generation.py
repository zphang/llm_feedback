import argparse
import tqdm.auto as tqdm
import os
import json

import llm_feedback.pilot.tasks as tasks
import llm_feedback.utils.env as env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_llm", type=str)
    parser.add_argument("--feedback_llm", type=str, default=None)
    parser.add_argument("--refinement_llm", type=str, default=None)
    parser.add_argument("--task", type=str)
    parser.add_argument("--task_args_str", type=str, default=None)
    parser.add_argument("--chain_name", default=None)  # If there are different langchains
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--max_num_examples", type=int, default=50)  # takes first n examples
    parser.add_argument("--output_dir")
    parser.add_argument("--run_batched", action="store_true",
                        help="Mainly needed for external tooling e.g. retrieval")
    args = parser.parse_args()
    if args.feedback_llm is None:
        args.feedback_llm = args.generation_llm
    if args.refinement_llm is None:
        args.refinement_llm = args.generation_llm

    env.load_dotenv()

    task = tasks.get_task(task_name=args.task, task_args_str=args.task_args_str)
    chain = task.get_chain(
        generation_llm=args.generation_llm,
        feedback_llm=args.feedback_llm,
        refinement_llm=args.refinement_llm,
        chain_name=args.chain_name
    )
    dataset = task.get_dataset(phase=args.phase)
    max_num_examples = min(args.max_num_examples, len(dataset))
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
        if args.task == "alfworld":
            # alfworld has a weird setup where the examples are actually environments
            # noinspection PyUnresolvedReferences
            all_outputs = task.process_all(chain=chain, dataset=dataset, max_num_examples=max_num_examples)
            for elem in all_outputs:
                f.write(json.dumps(elem) + "\n")
        elif args.run_batched:
            sub_dataset = [dataset[i] for i in range(max_num_examples)]
            all_outputs = task.batch_process(chain=chain, example_list=sub_dataset)
            for row in all_outputs:
                f.write(json.dumps(row) + "\n")
        else:
            for i, example in zip(tqdm.trange(max_num_examples), dataset):
                all_outputs = task.process(chain=chain, example=example)
                f.write(json.dumps(all_outputs) + "\n")

    print(f"Wrote {max_num_examples} {args.task} outputs to {write_path}.")


if __name__ == "__main__":
    main()
