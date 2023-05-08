#mkdir "./logs"
#mkdir "./models"
#mkdir "./cache"
#
#CUDA_VISIBLE_DEVICES=2 python main.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --model_type codet5 \
#    --task classification \
#    --num_labels 20 \
#    --sub_task ['none'] \
#    --output_dir="./models/codet5_12345" \
#    --data_dir="../dataset/" \
#    --cache_path="./cache" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval  --do_test  \
#    --train_batch_size 8 \
#    --eval_batch_size 16 \
#    --max_source_length 512 \
#    --num_train_epochs 20 \
#    --learning_rate 5e-5 \
#    --seed 12345 2>&1| tee ./logs/train_12345.log

#mkdir "./models/codet5_small"
#CUDA_VISIBLE_DEVICES=2 python main.py \
#    --model_name_or_path='Salesforce/codet5-small' \
#    --config_name='Salesforce/codet5-small' \
#    --tokenizer_name='Salesforce/codet5-small' \
#    --model_type codet5 \
#    --task classification \
#    --num_labels 20 \
#    --sub_task ['none'] \
#    --output_dir="./models/codet5_small" \
#    --data_dir="../dataset/" \
#    --cache_path="./cache" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_eval_bleu --do_test  \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 8 \
#    --eval_batch_size 16 \
#    --max_source_length 512 \
#    --num_train_epochs 20 \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/train_small.log

mkdir "./models/codet5_large_test"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --model_name_or_path='Salesforce/codet5-large' \
    --config_name='Salesforce/codet5-large' \
    --tokenizer_name='Salesforce/codet5-large' \
    --model_type codet5 \
    --task classification \
    --num_labels 20 \
    --sub_task ['none'] \
    --output_dir="./models/codet5_large_test" \
    --data_dir="../dataset/" \
    --cache_path="./cache" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_train --do_eval --do_test  \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --max_source_length 512 \
    --num_train_epochs 20 \
    --learning_rate 5e-5 \
    --seed 123456 2>&1| tee ./logs/train_large_test.log


    CUDA_VISIBLE_DEVICES=2,3 python main.py \
    --model_name_or_path='Salesforce/codet5-large' \
    --config_name='Salesforce/codet5-large' \
    --tokenizer_name='Salesforce/codet5-large' \
    --model_type codet5 \
    --task classification \
    --num_labels 20 \
    --sub_task ['none'] \
    --output_dir="./models/codet5_large_test" \
    --data_dir="../dataset/" \
    --cache_path="./cache" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_test  \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --max_source_length 512 \
    --num_train_epochs 20 \
    --learning_rate 5e-5 \
    --seed 123456


#### label smoothing ####
#mkdir "./models/codet5_ls_03"
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --model_type codet5 \
#    --task classification \
#    --num_labels 20 \
#    --sub_task ['none'] \
#    --output_dir="./models/codet5_ls_03" \
#    --data_dir="../dataset/" \
#    --cache_path="./cache" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_test  \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 8 \
#    --eval_batch_size 16 \
#    --max_source_length 512 \
#    --num_train_epochs 20 \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/train_ls_03.log