python llm_feedback/pilot/run_pilot_generation.py \
    --generation_llm llama \
    --feedback_llm gpt-4-0314 \
    --refinement_llm llama \
    --task mathqa \
    --phase train \
    --max_num_examples 50 \
    --output_dir ./outputs \
