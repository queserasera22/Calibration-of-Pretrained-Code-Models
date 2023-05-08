
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name=microsoft/graphcodebert-base \
#    --do_train \
#    --train_data_file=../dataset/CodeNet_Java250/OOD_token/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Java250/OOD_token/valid.jsonl \
#    --output_dir=./models/graphcodebert_OOD_token \
#    --dataset=Java250 \
#    --num_labels 250 \
#    --epoch 10 \
#    --block_size 512 \
#    --train_batch_size 32 \
#    --eval_batch_size 128 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_graphcodebert_Java250_OOD_token.log
#
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name=microsoft/graphcodebert-base \
#    --do_eval \
#    --train_data_file=../dataset/CodeNet_Java250/OOD_token/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Java250/OOD_token/test.jsonl \
#    --output_dir=./models/graphcodebert_OOD_token \
#    --dataset=Java250 \
#    --num_labels 250 \
#    --block_size 512 \
#    --eval_batch_size 128 \
#    --seed 123456 2>&1| tee ./logs/eval_graphcodebert_Java250_OOD_token.log
#
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name=microsoft/graphcodebert-base \
#    --do_train \
#    --train_data_file=../dataset/CodeNet_Java250/OOD_CST/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Java250/OOD_CST/valid.jsonl \
#    --output_dir=./models/graphcodebert_OOD_CST \
#    --dataset=Java250 \
#    --num_labels 250 \
#    --epoch 10 \
#    --block_size 512 \
#    --train_batch_size 32 \
#    --eval_batch_size 128 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_graphcodebert_Java250_OOD_CST.log
#
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --model_name=microsoft/graphcodebert-base \
#    --do_eval \
#    --train_data_file=../dataset/CodeNet_Java250/OOD_CST/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Java250/OOD_CST/test.jsonl \
#    --output_dir=./models/graphcodebert_OOD_CST \
#    --dataset=Java250 \
#    --num_labels 250 \
#    --block_size 512 \
#    --eval_batch_size 128 \
#    --seed 123456 2>&1| tee ./logs/eval_graphcodebert_Java250_OOD_CST.log



CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/CodeNet_Java250/OOD/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD/valid.jsonl \
    --output_dir=./models/graphcodebert_OOD_test \
    --dataset=Java250 \
    --num_labels 250 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 12345 2>&1| tee ./logs/train_graphcodebert_Java250_OOD_test.log

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../dataset/CodeNet_Java250/OOD/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD/test.jsonl \
    --output_dir=./models/graphcodebert_OOD_test \
    --dataset=Java250 \
    --num_labels 250 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 12345 2>&1| tee ./logs/eval_graphcodebert_Java250_OOD_test.log
