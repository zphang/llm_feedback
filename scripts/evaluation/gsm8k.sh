# get the file name
filename=$(basename $0)
# remove the file extension
task=${filename%.*}
# output dir the the 
python llm_feedback/pilot/run_pilot_evaluation.py \
    --model_outputs_path ./outputs/gpt-3.5-turbo-0301__gpt-3.5-turbo-0301__gpt-3.5-turbo-0301__${task}__train__outputs.jsonl \
    --task $task \
    --output_dir .