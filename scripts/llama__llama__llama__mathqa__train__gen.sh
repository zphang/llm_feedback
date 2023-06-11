python llm_feedback/pilot/run_pilot_generation.py \
    --generation_llm llama \
    --feedback_llm llama \
    --refinement_llm llama \
    --task mathqa \
    --phase train \
    --max_num_examples 50 \
    --output_dir ./outputs \
