#CUDA_VISIBLE_DEVICES=2 python main.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --model_type codet5 \
#    --task classification \
#    --sub_task ['none'] \
#    --output_dir="./models/POJ104" \
#    --data_dir="../dataset/POJ104" \
#    --num_labels 104 \
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
#    --seed 123456 2>&1| tee ./logs/train_POJ104_best_acc.log

mkdir "models/codet5_small/POJ104"
mkdir "cache/codet5_small"
CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name_or_path='Salesforce/codet5-small' \
    --config_name='Salesforce/codet5-small' \
    --tokenizer_name='Salesforce/codet5-small' \
    --model_type codet5 \
    --task classification \
    --sub_task ['none'] \
    --output_dir="./models/codet5_small/POJ104" \
    --data_dir="../dataset/POJ104" \
    --num_labels 104 \
    --cache_path="./cache/codet5_small" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_train --do_eval --do_test  \
    --save_last_checkpoints --always_save_model \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --max_source_length 512 \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --seed 123456 2>&1| tee ./logs/train_POJ104_small.log




