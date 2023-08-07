#!/usr/bin/env bash
generation_llm=gpt-3.5-turbo-0301
feedback_llm=claude-2
refinement_llm=gpt-3.5-turbo-0301
task=hotpotqa
python llm_feedback/pilot/run_pilot_generation.py \
    --generation_llm ${generation_llm} \
    --feedback_llm ${feedback_llm} \
    --refinement_llm ${refinement_llm} \
    --task ${task} \
    --max_num_examples 50 \
    --output_dir ./outputs

python llm_feedback/pilot/run_pilot_evaluation.py \
    --model_outputs_path ./outputs/${generation_llm}__${feedback_llm}__${refinement_llm}__${task}__train__outputs.jsonl \
    --task ${task} \
    --output_dir .
# task=mathqa
# python llm_feedback/pilot/run_pilot_generation.py \
#     --generation_llm ${generation_llm} \
#     --feedback_llm ${feedback_llm} \
#     --refinement_llm ${refinement_llm} \
#     --task ${task} \
#     --max_num_examples 50 \
#     --output_dir ./outputs