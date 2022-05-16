
# python ./src/inference.py\
#  --model_name_or_path "./output_roberta"\
#  --output_dir "/opt/ml/input/code/prediction_roberta_top_k"\
#  --dataset_name "/opt/ml/input/data/test_dataset/"\
#  --max_answer_length 100\
#  --use_faiss False\
#  --overwrite_output_dir True\
#  --do_predict\

python ./src/inference.py\
 --model_name_or_path "../models/output/checkpoint-300" \
 --output_dir "../predictions/prediction"\
 --dataset_name "../data/test_dataset"\
 --retrieval "DPR"\
 --fp16 \
 --top_k_retrieval 20\
 --max_answer_length 100\
 --use_faiss False\
 --overwrite_output_dir True\
 --do_predict\
