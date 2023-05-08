mkdir logs

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/POJ104/train.jsonl \
    --eval_data_file=../dataset/POJ104/valid.jsonl \
    --output_dir=./models/codebert \
    --dataset=POJ104 \
    --num_labels 104 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_POJ104.log

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_name=microsoft/codebert-base \
    --do_eval \
    --train_data_file=../dataset/POJ104/train.jsonl \
    --eval_data_file=../dataset/POJ104/test.jsonl \
    --output_dir=./models/codebert \
    --dataset=POJ104 \
    --num_labels 104 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 123456 2>&1| tee ../logs/eval_POJ104.log

