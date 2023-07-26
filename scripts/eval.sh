#!/usr/bin/env bash
# generation_llm=mergered_sl-alex_llama-7b-alpaca-stepwise-lora
# feedback_llm=mergered_sl-alex_llama-7b-alpaca-stepwise-lora
# refinement_llm=mergered_sl-alex_llama-7b-alpaca-stepwise-lora
generation_llm=gpt-4-0314
feedback_llm=gpt-3.5-turbo-0301
refinement_llm=gpt-4-0314
task=gsm8k
python llm_feedback/pilot/run_pilot_evaluation.py \
    --model_outputs_path ./outputs/${generation_llm}__${feedback_llm}__${refinement_llm}__${task}__train__outputs.jsonl \
    --task ${task} \
    --output_dir .

# generation_llm=gpt-4-031
# feedback_llm=mergered_sl-alex_llama-7b-alpaca-stepwise-lora
# refinement_llm=gpt-4-031
# task=gsm8k
# python llm_feedback/pilot/run_pilot_generation.py \
#     --generation_llm ${generation_llm} \
#     --feedback_llm ${feedback_llm} \
#     --refinement_llm ${refinement_llm} \
#     --task ${task} \
#     --max_num_examples 50 \
#     --output_dir ./outputs
