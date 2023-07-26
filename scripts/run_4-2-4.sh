#!/usr/bin/env bash
generation_llm=gpt-4-0314
feedback_llm=Llama-2-70b-chat-hf
refinement_llm=gpt-4-0314
task=mathqa
python llm_feedback/pilot/run_pilot_generation.py \
    --generation_llm ${generation_llm} \
    --feedback_llm ${feedback_llm} \
    --refinement_llm ${refinement_llm} \
    --task ${task} \
    --max_num_examples 50 \
    --output_dir ./outputs

# task=mathqa
# python llm_feedback/pilot/run_pilot_generation.py \
#     --generation_llm ${generation_llm} \
#     --feedback_llm ${feedback_llm} \
#     --refinement_llm ${refinement_llm} \
#     --task ${task} \
#     --max_num_examples 50 \
#     --output_dir ./outputs