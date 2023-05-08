mkdir "models/codet5_large/POJ104"
CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name_or_path='Salesforce/codet5-large' \
    --config_name='Salesforce/codet5-large' \
    --tokenizer_name='Salesforce/codet5-large' \
    --model_type codet5 \
    --task classification \
    --sub_task ['none'] \
    --output_dir="./models/codet5_large/POJ104" \
    --data_dir="../dataset/POJ104" \
    --num_labels 104 \
    --cache_path="./cache" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_train --do_eval --do_test  \
    --train_batch_size 2 \
    --eval_batch_size 16 \
    --max_source_length 512 \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --seed 123456 2>&1| tee ./logs/train_POJ104_large.log

mkdir "models/codet5_large/Java250_12345"
CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name_or_path='Salesforce/codet5-large' \
    --config_name='Salesforce/codet5-large' \
    --tokenizer_name='Salesforce/codet5-large' \
    --model_type codet5 \
    --task classification \
    --sub_task ['none'] \
    --output_dir="./models/codet5_large/Java250_12345" \
    --data_dir="../dataset/CodeNet_Java250" \
    --num_labels 250 \
    --cache_path="./cache" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_train --do_eval --do_test  \
    --train_batch_size 2 \
    --eval_batch_size 16 \
    --max_source_length 512 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --seed 12345 2>&1| tee ./logs/train_Java250_large_12345.log

mkdir "models/codet5_large/Python800_test"
CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name_or_path='Salesforce/codet5-large' \
    --config_name='Salesforce/codet5-large' \
    --tokenizer_name='Salesforce/codet5-large' \
    --model_type codet5 \
    --task classification \
    --sub_task ['none'] \
    --output_dir="./models/codet5_large/Python800_test" \
    --data_dir="../dataset/CodeNet_Python800" \
    --num_labels 800 \
    --cache_path="./cache" \
    --res_dir="./logs" \
    --summary_dir="./logs" \
    --do_train --do_eval --do_test  \
    --train_batch_size 2 \
    --eval_batch_size 16 \
    --max_source_length 512 \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --seed 123456 2>&1| tee ./logs/train_Python800_large_test.log