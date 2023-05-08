CUDA_VISIBLE_DEVICES=1 python main.py \
    --do_train \
    --model_name=microsoft/graphcodebert-base \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/graphcodebert \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_graphcodebert.log

CUDA_VISIBLE_DEVICES=1 python main.py \
    --do_eval \
    --model_name=microsoft/graphcodebert-base \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/graphcodebert \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456 2>&1| tee ./logs/eval_graphcodebert.log