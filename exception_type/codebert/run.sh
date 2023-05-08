CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/graphcodebert_ls_01 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --label_smoothing 0.1 \
    --seed 123456 2>&1| tee ./logs/train_graphcodebert_ls_01.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/graphcodebert_ls_01 \
    --block_size 512 \
    --eval_batch_size 128 \
    --label_smoothing 0.1 \
    --seed 123456 2>&1| tee ./logs/eval_graphcodebert_ls_01.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/graphcodebert_ls_02 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --label_smoothing 0.2 \
    --seed 123456 2>&1| tee ./logs/train_graphcodebert_ls_02.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/graphcodebert_ls_02 \
    --block_size 512 \
    --eval_batch_size 128 \
    --label_smoothing 0.2 \
    --seed 123456 2>&1| tee ./logs/eval_graphcodebert_ls_02.log


CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/graphcodebert_ls_03 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --label_smoothing 0.3 \
    --seed 123456 2>&1| tee ./logs/train_graphcodebert_ls_03.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/graphcodebert_ls_03 \
    --block_size 512 \
    --eval_batch_size 128 \
    --label_smoothing 0.3 \
    --seed 123456 2>&1| tee ./logs/eval_graphcodebert_ls_03.log