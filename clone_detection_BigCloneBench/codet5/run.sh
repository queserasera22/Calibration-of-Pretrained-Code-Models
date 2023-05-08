#CUDA_VISIBLE_DEVICES=1 python main.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --model_type codet5 \
#    --task clone \
#    --sub_task ['none'] \
#    --output_dir="./models" \
#    --data_dir="../dataset" \
#    --cache_path="./cache" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_eval_bleu  --do_test  \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 4 \
#    --eval_batch_size 8 \
#    --max_source_length 512 \
#    --num_train_epochs 1 \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/test.log


#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name_or_path='Salesforce/codet5-small' \
#    --config_name='Salesforce/codet5-small' \
#    --tokenizer_name='Salesforce/codet5-small' \
#    --model_type codet5 \
#    --task clone \
#    --sub_task ['none'] \
#    --output_dir="./models/codet5_small" \
#    --data_dir="../dataset" \
#    --cache_path="./cache/codet5_small" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_eval_bleu  --do_test  \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 4 \
#    --eval_batch_size 8 \
#    --max_source_length 512 \
#    --num_train_epochs 1 \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/test_small.log


#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name_or_path='Salesforce/codet5-large' \
#    --config_name='Salesforce/codet5-large' \
#    --tokenizer_name='Salesforce/codet5-large' \
#    --model_type codet5 \
#    --task clone \
#    --sub_task ['none'] \
#    --output_dir="./models/codet5_large" \
#    --data_dir="../dataset" \
#    --cache_path="./cache/codet5_large" \
#    --res_dir="./logs" \
#    --summary_dir="./logs" \
#    --do_train --do_eval --do_eval_bleu  --do_test  \
#    --save_last_checkpoints --always_save_model \
#    --train_batch_size 1 \
#    --eval_batch_size 4 \
#    --max_source_length 512 \
#    --num_train_epochs 4 \
#    --start_epoch 3 \
#    --load_model_path="./models/codet5_large/checkpoint-best-f1/pytorch_model.bin" \
#    --learning_rate 5e-5 \
#    --seed 123456 2>&1| tee ./logs/train_large.log

##### train codet5-base with label smoothing #####
mkdir "./models/codet5_base_ls_03"
CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --model_type codet5 \
    --task clone \
    --sub_task ['none'] \
    --output_dir="./models/codet5_base_ls_03" \
    --data_dir="../dataset" \
    --cache_path="./cache" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_train --do_eval  --do_test  \
    --save_last_checkpoints --always_save_model \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --max_source_length 512 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --label_smoothing 0.3 \
    --seed 123456 2>&1| tee ./logs/train_codet5_base_ls_03.log

