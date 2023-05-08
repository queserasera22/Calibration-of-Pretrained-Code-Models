#mkdir logs
#
#CUDA_VISIBLE_DEVICES=2 python main.py \
#    --model_name=huggingface/CodeBERTa-small-v1 \
#    --do_train \
#    --train_data_file=../dataset/CodeNet_Java250/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Java250/valid.jsonl \
#    --output_dir=./models/codeberta \
#    --dataset=Java250 \
#    --num_labels 250 \
#    --epoch 10 \
#    --block_size 512 \
#    --train_batch_size 48 \
#    --eval_batch_size 128 \
#    --learning_rate 1e-4 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_codeberta_Java250.log
#
#CUDA_VISIBLE_DEVICES=1 python main.py \
#    --model_name=huggingface/CodeBERTa-small-v1 \
#    --do_eval \
#    --train_data_file=../dataset/CodeNet_Java250/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Java250/test.jsonl \
#    --output_dir=./models/codeberta \
#    --dataset=Java250 \
#    --num_labels 250 \
#    --block_size 512 \
#    --eval_batch_size 128 \
#    --seed 123456 2>&1| tee ../logs/eval_codeberta_Java250.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=huggingface/CodeBERTa-small-v1 \
    --do_train \
    --train_data_file=../dataset/CodeNet_Java250/OOD_CST/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_CST/valid.jsonl \
    --output_dir=./models/codeberta_OOD_CST/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 48 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_codeberta_Java250_OOD_CST_ls_005.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=huggingface/CodeBERTa-small-v1 \
    --do_eval \
    --train_data_file=../dataset/CodeNet_Java250/OOD_CST/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_CST/test.jsonl \
    --output_dir=./models/codeberta_OOD_CST/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 123456 2>&1| tee ./logs/eval_codeberta_Java250_OOD_CST_ls_005.log


CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/CodeNet_Java250/OOD_CST/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_CST/valid.jsonl \
    --output_dir=./models/codebert_OOD_CST/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_Java250_OOD_CST_ls_005.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=microsoft/codebert-base \
    --do_eval \
    --train_data_file=../dataset/CodeNet_Java250/OOD_CST/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_CST/test.jsonl \
    --output_dir=./models/codebert_OOD_CST/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 123456 2>&1| tee ./logs/eval_Java250_OOD_CST_ls_005.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=huggingface/CodeBERTa-small-v1 \
    --do_train \
    --train_data_file=../dataset/CodeNet_Java250/OOD_token/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_token/valid.jsonl \
    --output_dir=./models/codeberta_OOD_token/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 48 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_codeberta_Java250_OOD_token_ls_005.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=huggingface/CodeBERTa-small-v1 \
    --do_eval \
    --train_data_file=../dataset/CodeNet_Java250/OOD_token/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_token/test.jsonl \
    --output_dir=./models/codeberta_OOD_token/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 123456 2>&1| tee ./logs/eval_codeberta_Java250_OOD_token_ls_005.log


CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/CodeNet_Java250/OOD_token/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_token/valid.jsonl \
    --output_dir=./models/codebert_OOD_token/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_Java250_OOD_token_ls_005.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name=microsoft/codebert-base \
    --do_eval \
    --train_data_file=../dataset/CodeNet_Java250/OOD_token/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_token/test.jsonl \
    --output_dir=./models/codebert_OOD_token/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 123456 2>&1| tee ./logs/eval_Java250_OOD_token_ls_005.log