#mkdir logs
#
CUDA_VISIBLE_DEVICES=2 python main.py \
    --do_train \
    --model_name=microsoft/codebert-base \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/codebert/test123 \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123 2>&1| tee ./logs/train_codebert_test123.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --do_eval \
    --model_name=microsoft/codebert-base \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/codebert/test123 \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123 2>&1| tee ./logs/eval_codebert_test123.log


#CUDA_VISIBLE_DEVICES=1 python main.py \
#    --do_train \
#    --model_name=microsoft/codebert-base \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/valid.jsonl \
#    --output_dir=./models/codebert/ls_03 \
#    --epoch 5 \
#    --block_size 512 \
#    --train_batch_size 16 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_codebert_ls_03.log
#
#CUDA_VISIBLE_DEVICES=1 python main.py \
#    --do_eval \
#    --model_name=microsoft/codebert-base \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/test.jsonl \
#    --output_dir=./models/codebert/ls_03 \
#    --block_size 512 \
#    --eval_batch_size 64 \
#    --seed 123456 2>&1| tee ./logs/eval_codebert_ls_03.log
#
#
#CUDA_VISIBLE_DEVICES=1 python main.py \
#    --do_train \
#    --model_name=huggingface/CodeBERTa-small-v1 \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/valid.jsonl \
#    --output_dir=./models/codeberta/ls_03 \
#    --epoch 5 \
#    --block_size 512 \
#    --train_batch_size 32 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_codeberta_ls_03.log
#
#CUDA_VISIBLE_DEVICES=1 python main.py \
#    --do_eval \
#    --model_name=huggingface/CodeBERTa-small-v1 \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/test.jsonl \
#    --output_dir=./models/codeberta/ls_03 \
#    --block_size 512 \
#    --eval_batch_size 64 \
#    --seed 123456 2>&1| tee ./logs/eval_codeberta_ls_03.log