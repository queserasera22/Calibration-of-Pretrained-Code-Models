#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --model_type codet5 \
#    --task classification \
#    --sub_task ['none'] \
#    --output_dir="./models/Java250" \
#    --data_dir="../dataset/CodeNet_Java250" \
#    --num_labels 250 \
#    --cache_path="./cache" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_test  \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 8 \
#    --eval_batch_size 16 \
#    --max_source_length 512 \
#    --num_train_epochs 2 \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/train_Java250_best_acc.log

#
#mkdir "models/codet5_small/Java250"
#mkdir "cache/codet5_small"
#CUDA_VISIBLE_DEVICES=2 python main.py \
#    --model_name_or_path='Salesforce/codet5-small' \
#    --config_name='Salesforce/codet5-small' \
#    --tokenizer_name='Salesforce/codet5-small' \
#    --model_type codet5 \
#    --task classification \
#    --sub_task ['none'] \
#    --output_dir="./models/codet5_small/Java250" \
#    --data_dir="../dataset/CodeNet_Java250" \
#    --num_labels 250 \
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
#    --seed 123456 2>&1| tee ./logs/train_Java250_small.log

#mkdir "models/codet5_small/Java250"
#mkdir "cache/codet5_small"
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --model_type codet5 \
#    --task classification \
#    --sub_task ['none'] \
#    --output_dir="./models/Java250" \
#    --data_dir="../dataset/CodeNet_Java250" \
#    --num_labels 250 \
#    --cache_path="./cache" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_test  \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 8 \
#    --eval_batch_size 16 \
#    --max_source_length 512 \
#    --num_train_epochs 2 \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/train_Java250_best_acc.log


mkdir "models/codet5_OOD_token"
mkdir "models/codet5_OOD_token/Java250"
mkdir "models/codet5_OOD_token/Java250/ls_005"
mkdir "cache/codet5_OOD_token"
CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --model_type codet5 \
    --task classification \
    --sub_task ['none'] \
    --output_dir="./models/codet5_OOD_token/Java250/ls_005" \
    --data_dir="../dataset/CodeNet_Java250/OOD_token/" \
    --num_labels 250 \
    --cache_path="./cache/codet5_OOD_token" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_train --do_eval --do_test \
    --always_save_model \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --max_source_length 512 \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --seed 123456 2>&1| tee ./logs/train_Java250_OOD_token_ls_005.log


mkdir "models/codet5_OOD_CST"
mkdir "models/codet5_OOD_CST/Java250"
mkdir "models/codet5_OOD_CST/Java250/ls_005"
mkdir "cache/codet5_OOD_CST"
CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --model_type codet5 \
    --task classification \
    --sub_task ['none'] \
    --output_dir="./models/codet5_OOD_CST/Java250/ls_005" \
    --data_dir="../dataset/CodeNet_Java250/OOD_CST/" \
    --num_labels 250 \
    --cache_path="./cache/codet5_OOD_CST" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_train --do_eval --do_test \
    --always_save_model \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --max_source_length 512 \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --seed 123456 2>&1| tee ./logs/train_Java250_OOD_CST_ls_005.log

mkdir "models/codet5_OOD"
mkdir "models/codet5_OOD/Java250"
mkdir "models/codet5_OOD/Java250/ls_005"
mkdir "cache/codet5_OOD"
CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --model_type codet5 \
    --task classification \
    --sub_task ['none'] \
    --output_dir="./models/codet5_OOD/Java250/ls_005" \
    --data_dir="../dataset/CodeNet_Java250/OOD/" \
    --num_labels 250 \
    --cache_path="./cache/codet5_OOD" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_train --do_eval --do_test \
    --always_save_model \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --max_source_length 512 \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --seed 123456 2>&1| tee ./logs/train_Java250_OOD_ls_005.log

cd ../codebert
sh run_codebert_java250.sh