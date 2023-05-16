# get the file name
filename=$(basename $0)
# remove the file extension
task=${filename%.*}
python llm_feedback/pilot/run_pilot_generation.py \
    --generation_llm gpt-3.5-turbo-0301 \
    --task $task \
    --phase train \
    --max_num_examples 50 \
    --output_dir ./outputs