CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/graphcodebert_test \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 12345 2>&1| tee ./logs/train_graphcodebert_test.log

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/graphcodebert_test \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 12345 2>&1| tee ./logs/eval_graphcodebert_test.log

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/graphcodebert_1234 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 1234 2>&1| tee ./logs/train_graphcodebert_1234.log

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/graphcodebert_1234 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 1234 2>&1| tee ./logs/eval_graphcodebert_1234.log

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/graphcodebert_123 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123 2>&1| tee ./logs/train_graphcodebert_123.log

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/graphcodebert_123 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 123 2>&1| tee ./logs/eval_graphcodebert_123.log
