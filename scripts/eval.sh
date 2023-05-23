generation_llm=vicuna-13b
feedback_llm=vicuna-13b
refinement_llm=vicuna-13b
task=mathqa
python llm_feedback/pilot/run_pilot_evaluation.py \
    --model_outputs_path ./outputs/${generation_llm}__${feedback_llm}__${refinement_llm}__${task}__train__outputs.jsonl \
    --task ${task} \
    --output_dir .