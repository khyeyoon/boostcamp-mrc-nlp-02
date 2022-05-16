
# python ./src/train.py\
#  --model_name_or_path "klue/roberta-large"\
#  --output_dir "output_train_with_bm25"\
# --overwrite_output_dir True\

# --do_train\
# --do_eval

python ./src/train.py \
--output_dir "../models/output" \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--eval_steps 300 --save_strategy steps --save_steps 300 \
--evaluation_strategy  steps \
--model_name_or_path "klue/bert-base" \
--num_train_epochs 1 \
--save_total_limit 3 \
--greater_is_better True \
--metric_for_best_model exact_match \
--overwrite_output_dir False \
--fp16 True \
--load_best_model_at_end True --do_train --do_eval