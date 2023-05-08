#CUDA_VISIBLE_DEVICES=2 python main.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --model_type codet5 \
#    --task classification \
#    --sub_task ['none'] \
#    --output_dir="./models/Python800" \
#    --data_dir="../dataset/CodeNet_Python800" \
#    --num_labels 800 \
#    --cache_path="./cache" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_eval_bleu --do_test  \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 8 \
#    --eval_batch_size 16 \
#    --max_source_length 512 \
#    --num_train_epochs 3 \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/train_Python800_best_acc.log

#mkdir "models/codet5_small/Python800"
#mkdir "cache/codet5_small"
#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name_or_path='Salesforce/codet5-small' \
#    --config_name='Salesforce/codet5-small' \
#    --tokenizer_name='Salesforce/codet5-small' \
#    --model_type codet5 \
#    --task classification \
#    --sub_task ['none'] \
#    --output_dir="./models/codet5_small/Python800" \
#    --data_dir="../dataset/CodeNet_Python800" \
#    --num_labels 800 \
#    --cache_path="./cache/codet5_small" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_test  \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 8 \
#    --eval_batch_size 16 \
#    --max_source_length 512 \
#    --num_train_epochs 3 \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/train_Python800_small.log

##
##mkdir "models/codet5_OOD_token"
##mkdir "models/codet5_OOD_token/Python800"
##mkdir "models/codet5_OOD_token/Python800/ls_01"
##mkdir "cache/codet5_OOD_token"
#CUDA_VISIBLE_DEVICES=2 python main.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --model_type codet5 \
#    --task classification \
#    --dataset_name "Python800" \
#    --sub_task ['none'] \
#    --output_dir="./models/codet5_OOD_token/Python800" \
#    --data_dir="../dataset/CodeNet_Python800/OOD_token/" \
#    --num_labels 800 \
#    --cache_path="./cache/codet5_OOD_token" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_test \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 8 \
#    --eval_batch_size 16 \
#    --max_source_length 512 \
#    --num_train_epochs 3 \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/train_Python800_OOD_token.log


#mkdir "models/codet5_OOD_CST"
#mkdir "models/codet5_OOD_CST/Python800"
#mkdir "models/codet5_OOD_CST/Python800/ls_01"
#mkdir "cache/codet5_OOD_CST"
#CUDA_VISIBLE_DEVICES=2 python main.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --model_type codet5 \
#    --task classification \
#    --dataset_name "Python800" \
#    --sub_task ['none'] \
#    --output_dir="./models/codet5_OOD_CST/Python800/ls_01" \
#    --data_dir="../dataset/CodeNet_Python800/OOD_CST/" \
#    --num_labels 800 \
#    --cache_path="./cache/codet5_OOD_CST" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_test \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 8 \
#    --eval_batch_size 16 \
#    --max_source_length 512 \
#    --num_train_epochs 3 \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/train_Python800_OOD_CST_ls_01.log



mkdir "models/codet5_OOD"
mkdir "models/codet5_OOD/Python800"
mkdir "models/codet5_OOD/Python800/ls_01"
mkdir "cache/codet5_OOD"
CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --model_type codet5 \
    --task classification \
    --dataset_name "Python800" \
    --sub_task ['none'] \
    --output_dir="./models/codet5_OOD/Python800/ls_01" \
    --data_dir="../dataset/CodeNet_Python800/OOD/" \
    --num_labels 800 \
    --cache_path="./cache/codet5_OOD" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_train --do_eval --do_test \
    --save_last_checkpoints --always_save_model \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --max_source_length 512 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --seed 123456 2>&1| tee ./logs/train_Python800_OOD_ls_01.log