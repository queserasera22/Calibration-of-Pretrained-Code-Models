#mkdir logs
#
CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/codebert_test \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_codebert_test.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=microsoft/codebert-base \
    --do_eval \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/codebert_test \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 123456 2>&1| tee ./logs/eval_codebert_test.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/codebert_test1 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 12345 2>&1| tee ./logs/train_codebert_test1.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=microsoft/codebert-base \
    --do_eval \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/codebert_test1 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 12345 2>&1| tee ./logs/eval_codebert_test1.log

#######  label smoothing  ######
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name=microsoft/codebert-base \
#    --do_train \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/valid.jsonl \
#    --output_dir=./models/codebert_ls_03 \
#    --epoch 10 \
#    --block_size 512 \
#    --train_batch_size 32 \
#    --eval_batch_size 128 \
#    --learning_rate 1e-4 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_codebert_ls_03.log
#
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name=microsoft/codebert-base \
#    --do_eval \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/test.jsonl \
#    --output_dir=./models/codebert_ls_03 \
#    --block_size 512 \
#    --eval_batch_size 128 \
#    --seed 123456 2>&1| tee ./logs/eval_codebert_ls_03.log
#
#
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name=huggingface/CodeBERTa-small-v1 \
#    --do_train \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/valid.jsonl \
#    --output_dir=./models/codeberta_ls_03 \
#    --epoch 10 \
#    --block_size 512 \
#    --train_batch_size 48 \
#    --eval_batch_size 128 \
#    --learning_rate 1e-4 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_codeberta_ls_03.log
#
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name=huggingface/CodeBERTa-small-v1 \
#    --do_eval \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/test.jsonl \
#    --output_dir=./models/codeberta_ls_03 \
#    --block_size 512 \
#    --eval_batch_size 128 \
#    --seed 123456 2>&1| tee ./logs/eval_codeberta_ls_03.log