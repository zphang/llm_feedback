python llm_feedback/pilot/run_pilot_generation.py \
    --generation_llm gpt-3.5-turbo-0301 \
    --feedback_llm gpt-3.5-turbo-0301 \
    --refinement_llm llama \
    --task mathqa \
    --phase train \
    --max_num_examples 50 \
    --output_dir ./outputs \
