generation_llm=vicuna-13b
feedback_llm=vicuna-13b
refinement_llm=vicuna-13b
task=mathqa
python llm_feedback/pilot/run_pilot_generation.py \
    --generation_llm ${generation_llm} \
    --feedback_llm ${feedback_llm} \
    --refinement_llm ${refinement_llm} \
    --task ${task} \
    --max_num_examples 50 \
    --output_dir ./outputs
