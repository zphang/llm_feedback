python llm_feedback/pilot/run_pilot_generation.py \
    --generation_llm ggml-gpt4all-j \
    --feedback_llm gpt-4-0314 \
    --refinement_llm ggml-gpt4all-j \
    --task mathqa \
    --phase train \
    --max_num_examples 50 \
    --output_dir ./outputs \
