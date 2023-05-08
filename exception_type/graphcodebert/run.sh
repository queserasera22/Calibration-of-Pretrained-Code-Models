#mkdir "logs"
#
#CUDA_VISIBLE_DEVICES=0 python3 main.py \
#    --output_dir=./models/test_c \
#    --model_type=roberta \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --do_train \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/valid.jsonl \
#    --epoch 20 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 32 \
#    --eval_batch_size 64 \
#    --learning_rate 5e-5 \
#    --max_grad_norm 1.0 \
#    --warmup_steps 1000 \
#    --evaluate_during_training \
#    --seed 123456  2>&1 | tee ./logs/train_test_c.log
#
#CUDA_VISIBLE_DEVICES=2 python3 main.py \
#    --output_dir=./models/test_c \
#    --model_type=roberta \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --do_test \
#    --train_data_file=../dataset/train.jsonl \
#    --test_data_file=../dataset/test.jsonl \
#    --epoch 20 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 32 \
#    --eval_batch_size 64 \
#    --learning_rate 5e-5 \
#    --warmup_steps 1000 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1 | tee ./logs/eval_test_c.log



#### label smoothing ####
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --output_dir=./models/ls_03 \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --lang python \
    --epoch 20 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --warmup_steps 1000 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ./logs/train_ls_03.log

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --output_dir=./models/ls_03 \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --lang python \
    --epoch 20 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee ./logs/eval_ls_03.log
