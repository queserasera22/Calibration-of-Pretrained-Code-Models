#CUDA_VISIBLE_DEVICES=3 python3 main.py \
#    --output_dir=./models \
#    --model_type=roberta \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --do_train \
#    --train_data_file=../dataset/CodeNet_Java250/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Java250/valid.jsonl \
#    --test_data_file=../dataset/CodeNet_Java250/test.jsonl \
#    --dataset=Java250 \
#    --lang=java \
#    --num_labels 250 \
#    --epoch 20 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 32 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456  2>&1 | tee ./logs/train_java250.log
#
#CUDA_VISIBLE_DEVICES=1 python3 main.py \
#    --output_dir=./models \
#    --model_type=roberta \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --do_test \
#    --train_data_file=../dataset/CodeNet_Java250/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Java250/test.jsonl \
#    --test_data_file=../dataset/CodeNet_Java250/test.jsonl \
#    --dataset=Java250 \
#    --lang=java \
#    --num_labels 250 \
#    --epoch 20 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 32 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1 | tee ./logs/eval_java250.log

#mkdir "logs"
########## run codebert on Java250_OOD_token dataset ######
CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --output_dir=./models/OOD_token/ls_005 \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --do_test \
    --train_data_file=../dataset/CodeNet_Java250/OOD_token/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_token/valid.jsonl \
    --test_data_file=../dataset/CodeNet_Java250/OOD_token/test.jsonl \
    --dataset=Java250 \
    --lang=java \
    --num_labels 250 \
    --epoch 20 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ./logs/train_java250_OOD_token_ls_005.log

CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --output_dir=./models/OOD_CST/ls_005 \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --do_test \
    --train_data_file=../dataset/CodeNet_Java250/OOD_CST/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD_CST/valid.jsonl \
    --test_data_file=../dataset/CodeNet_Java250/OOD_CST/test.jsonl \
    --dataset=Java250 \
    --lang=java \
    --num_labels 250 \
    --epoch 20 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ./logs/train_java250_OOD_CST_ls_005.log


######### run codebert on Java250_OOD dataset ######
CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --output_dir=./models/OOD/ls_005 \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --do_test \
    --train_data_file=../dataset/CodeNet_Java250/OOD/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Java250/OOD/valid.jsonl \
    --test_data_file=../dataset/CodeNet_Java250/OOD/test.jsonl \
    --dataset=Java250 \
    --lang=java \
    --num_labels 250 \
    --epoch 20 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ./logs/train_java250_OOD_ls_005.log

