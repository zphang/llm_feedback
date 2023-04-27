# Learning from Feedback Experiments

## Setup

1. install the required packages

```bash
pip install -r requirements.txt
```

2. Set up a `.env` file in this folder. A template can be found at `.env_template`
3. Add to `PYTHONPATH`:

```bash
export PYTHONPATH=${PYTHONPATH}:/path/to/llm_feedback
```


## Running the experiments

Generating the outputs:

```bash
python llm_feedback/pilot/run_pilot_generation.py \
    --generation_llm gpt-3.5-turbo-0301 \
    --task example \
    --max_num_examples 50 \
    --output_dir /path/to/dir
```

Evaluating the outputs:

```bash
python llm_feedback/pilot/run_pilot_evaluation.py \
    --model_outputs_path /path/to/dir/gpt-3.5-turbo-0301__gpt-3.5-turbo-0301__gpt-3.5-turbo-0301__example__train__outputs.jsonl \
    --task example \
    --output_dir /path/to/dir
```

