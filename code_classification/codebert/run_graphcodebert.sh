### Train and evaluate on POJ104
#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name=microsoft/graphcodebert-base \
#    --do_train \
#    --train_data_file=../dataset/POJ104/train.jsonl \
#    --eval_data_file=../dataset/POJ104/valid.jsonl \
#    --output_dir=./models/graphcodebert \
#    --dataset=POJ104 \
#    --num_labels 104 \
#    --epoch 10 \
#    --block_size 512 \
#    --train_batch_size 32 \
#    --eval_batch_size 128 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_graphcodebert_POJ104.log
#
#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name=microsoft/graphcodebert-base \
#    --do_eval \
#    --train_data_file=../dataset/POJ104/train.jsonl \
#    --eval_data_file=../dataset/POJ104/test.jsonl \
#    --output_dir=./models/graphcodebert \
#    --dataset=POJ104 \
#    --num_labels 104 \
#    --block_size 512 \
#    --eval_batch_size 128 \
#    --seed 123456 2>&1| tee ../logs/eval_graphcodebert_POJ104.log
#
#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name=microsoft/graphcodebert-base \
#    --do_train \
#    --train_data_file=../dataset/CodeNet_Java250/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Java250/valid.jsonl \
#    --output_dir=./models/graphcodebert_12345 \
#    --dataset=Java250 \
#    --num_labels 250 \
#    --epoch 10 \
#    --block_size 512 \
#    --train_batch_size 32 \
#    --eval_batch_size 128 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 12345 2>&1| tee ./logs/train_graphcodebert_Java250_12345.log
#
#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name=microsoft/graphcodebert-base \
#    --do_eval \
#    --train_data_file=../dataset/CodeNet_Java250/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Java250/test.jsonl \
#    --output_dir=./models/graphcodebert_12345 \
#    --dataset=Java250 \
#    --num_labels 250 \
#    --block_size 512 \
#    --eval_batch_size 128 \
#    --seed 12345 2>&1| tee ../logs/eval_graphcodebert_Java250_12345.log
#
#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name=microsoft/graphcodebert-base \
#    --do_train \
#    --train_data_file=../dataset/CodeNet_Python800/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Python800/valid.jsonl \
#    --output_dir=./models/graphcodebert \
#    --dataset=Python800 \
#    --num_labels 800 \
#    --epoch 10 \
#    --block_size 512 \
#    --train_batch_size 32 \
#    --eval_batch_size 128 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_graphcodebert_Python800.log
#
#CUDA_VISIBLE_DEVICES=3 python main.py \
#    --model_name=microsoft/graphcodebert-base \
#    --do_eval \
#    --train_data_file=../dataset/CodeNet_Python800/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Python800/test.jsonl \
#    --output_dir=./models/graphcodebert \
#    --dataset=Python800 \
#    --num_labels 800 \
#    --block_size 512 \
#    --eval_batch_size 64 \
#    --seed 123456 2>&1| tee ../logs/eval_graphcodebert_Python800.log


CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/CodeNet_Java250/OOD_token/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_token/valid.jsonl \
    --output_dir=./models/graphcodebert_OOD_token/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --label_smoothing 0.05 \
    --seed 123456 2>&1| tee ./logs/train_graphcodebert_Java250_OOD_token_ls_005.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../dataset/CodeNet_Java250/OOD_token/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_token/test.jsonl \
    --output_dir=./models/graphcodebert_OOD_token/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --block_size 512 \
    --eval_batch_size 128 \
    --label_smoothing 0.05 \
    --seed 123456 2>&1| tee ./logs/eval_graphcodebert_Java250_OOD_token_ls_005.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/CodeNet_Java250/OOD_CST/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_CST/valid.jsonl \
    --output_dir=./models/graphcodebert_OOD_CST/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --label_smoothing 0.05 \
    --seed 123456 2>&1| tee ./logs/train_graphcodebert_Java250_OOD_CST_ls_005.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../dataset/CodeNet_Java250/OOD_CST/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_CST/test.jsonl \
    --output_dir=./models/graphcodebert_OOD_CST/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --block_size 512 \
    --eval_batch_size 128 \
    --label_smoothing 0.05 \
    --seed 123456 2>&1| tee ./logs/eval_graphcodebert_Java250_OOD_CST_ls_005.log



CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/CodeNet_Java250/OOD/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD/valid.jsonl \
    --output_dir=./models/graphcodebert_OOD/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --label_smoothing 0.05 \
    --seed 123456 2>&1| tee ./logs/train_graphcodebert_Java250_OOD_ls_005.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../dataset/CodeNet_Java250/OOD/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD/test.jsonl \
    --output_dir=./models/graphcodebert_OOD/ls_005 \
    --dataset=Java250 \
    --num_labels 250 \
    --block_size 512 \
    --eval_batch_size 128 \
    --label_smoothing 0.05 \
    --seed 123456 2>&1| tee ./logs/eval_graphcodebert_Java250_OOD_ls_005.log