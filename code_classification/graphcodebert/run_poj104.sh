mkdir "logs"

CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --output_dir=./models/test \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --do_test \
    --train_data_file=../dataset/POJ104/train.jsonl \
    --eval_data_file=../dataset/POJ104/valid.jsonl \
    --test_data_file=../dataset/POJ104/test.jsonl \
    --dataset=POJ104 \
    --num_labels 104 \
    --lang=c \
    --epoch 50 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ./logs/train_poj104_test.log