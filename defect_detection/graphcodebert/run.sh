#mkdir -p ./logs
#
#CUDA_VISIBLE_DEVICES=3 python3 main.py \
#    --output_dir=./models/test \
#    --model_type=roberta \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --do_train \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/valid.jsonl \
#    --test_data_file=../dataset/test.jsonl \
#    --epoch 10 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 16 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456  2>&1 | tee ./logs/train_test.log
#
#CUDA_VISIBLE_DEVICES=3 python3 main.py \
#    --output_dir=./models/test \
#    --model_type=roberta \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --do_test \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/test.jsonl \
#    --test_data_file=../dataset/test.jsonl \
#    --epoch 10 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 16 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1 | tee ./logs/eval_test.log


CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --output_dir=./models/ls_01_1234 \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 10 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --label_smoothing 0.1 \
    --seed 1234  2>&1 | tee ./logs/train_ls_01_1234.log