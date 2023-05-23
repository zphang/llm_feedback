generation_llm=llama-65b
feedback_llm=llama-65b
refinement_llm=llama-65b
task=gsm8k
python llm_feedback/pilot/run_pilot_generation.py \
    --generation_llm ${generation_llm} \
    --feedback_llm ${feedback_llm} \
    --refinement_llm ${refinement_llm} \
    --task ${task} \
    --max_num_examples 50 \
    --output_dir ./outputs
