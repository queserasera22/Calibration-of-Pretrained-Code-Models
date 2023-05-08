#CUDA_VISIBLE_DEVICES=2 python3 main.py \
#    --output_dir=./models \
#    --model_type=roberta \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --do_train \
#    --do_test \
#    --train_data_file=../dataset/CodeNet_Python800/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Python800/valid.jsonl \
#    --test_data_file=../dataset/CodeNet_Python800/test.jsonl \
#    --dataset=Python800 \
#    --lang=python \
#    --num_labels 800 \
#    --epoch 20 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 32 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456  2>&1 | tee ./logs/train_python800.log
#
#CUDA_VISIBLE_DEVICES=3 python3 main.py \
#    --output_dir=./models \
#    --model_type=roberta \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --do_test \
#    --train_data_file=../dataset/CodeNet_Python800/train.jsonl \
#    --test_data_file=../dataset/CodeNet_Python800/test.jsonl \
#    --dataset=Python800 \
#    --lang=python \
#    --num_labels 800 \
#    --epoch 20 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 32 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1 | tee ./logs/eval_python800.log


#CUDA_VISIBLE_DEVICES=1 python3 main.py \
#    --output_dir=./models/OOD_token/ls_01 \
#    --model_type=roberta \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --do_train \
#    --do_test \
#    --train_data_file=../dataset/CodeNet_Python800/OOD_token/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Python800/OOD_token/valid.jsonl \
#    --test_data_file=../dataset/CodeNet_Python800/OOD_token/test.jsonl \
#    --dataset=Python800 \
#    --lang=python \
#    --num_labels 800 \
#    --epoch 20 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 32 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1 | tee ./logs/train_python800_OOD_token_ls_01.log



#CUDA_VISIBLE_DEVICES=2 python3 main.py \
#    --output_dir=./models/OOD_CST/ls_01 \
#    --model_type=roberta \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --do_train \
#    --do_test \
#    --train_data_file=../dataset/CodeNet_Python800/OOD_CST/train.jsonl \
#    --eval_data_file=../dataset/CodeNet_Python800/OOD_CST/valid.jsonl \
#    --test_data_file=../dataset/CodeNet_Python800/OOD_CST/test.jsonl \
#    --dataset=Python800 \
#    --lang=python \
#    --num_labels 800 \
#    --epoch 20 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 32 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1 | tee ./logs/train_python800_OOD_CST_LS_01.log


CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --output_dir=./models/OOD/ls_01 \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --do_test \
    --train_data_file=../dataset/CodeNet_Python800/OOD/train.jsonl \
    --eval_data_file=../dataset/CodeNet_Python800/OOD/valid.jsonl \
    --test_data_file=../dataset/CodeNet_Python800/OOD/test.jsonl \
    --dataset=Python800 \
    --lang=python \
    --num_labels 800 \
    --epoch 20 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee ./logs/train_python800_OOD_ls_01.log

cd ../codebert
sh run_codeberta_java250.sh