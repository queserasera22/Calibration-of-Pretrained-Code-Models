#mkdir logs
#
#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name=huggingface/CodeBERTa-small-v1 \
#    --do_train \
#    --train_data_file=../dataset/CodeNet_Python800/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Python800/valid.jsonl \
#    --output_dir=./models/codeberta \
#    --dataset=Python800 \
#    --num_labels 800 \
#    --epoch 10 \
#    --block_size 512 \
#    --train_batch_size 48 \
#    --eval_batch_size 128 \
#    --learning_rate 1e-4 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_codeberta_Python800.log
#
#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name=huggingface/CodeBERTa-small-v1 \
#    --do_eval \
#    --train_data_file=../dataset/CodeNet_Python800/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Python800/test.jsonl \
#    --output_dir=./models/codeberta \
#    --dataset=Python800 \
#    --num_labels 800 \
#    --block_size 512 \
#    --eval_batch_size 128 \
#    --seed 123456 2>&1| tee ../logs/eval_codeberta_Python800.log


CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=huggingface/CodeBERTa-small-v1 \
    --do_train \
    --train_data_file=../dataset/CodeNet_Python800/OOD/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Python800/OOD/valid.jsonl \
    --output_dir=./models/codeberta_OOD \
    --dataset=Python800 \
    --num_labels 800 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 48 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_codeberta_Python800_OOD.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=huggingface/CodeBERTa-small-v1 \
    --do_eval \
    --train_data_file=../dataset/CodeNet_Python800/OOD/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Python800/OOD/test.jsonl \
    --output_dir=./models/codeberta_OOD \
    --dataset=Python800 \
    --num_labels 800 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 123456 2>&1| tee ./logs/eval_codeberta_Python800_OOD.log


CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/CodeNet_Python800/OOD/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Python800/OOD/valid.jsonl \
    --output_dir=./models/codebert_OOD \
    --dataset=Python800 \
    --num_labels 800 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_Python800_OOD.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/codebert-base \
    --do_eval \
    --train_data_file=../dataset/CodeNet_Python800/OOD/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Python800/OOD/test.jsonl \
    --output_dir=./models/codebert_OOD \
    --dataset=Python800 \
    --num_labels 800 \
    --block_size 512 \
    --eval_batch_size 128 \
    --seed 123456 2>&1| tee ./logs/eval_Python800_OOD.log

