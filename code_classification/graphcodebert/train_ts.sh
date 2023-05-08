#CUDA_VISIBLE_DEVICES=1 python3 train_ts.py \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --model_path=./models/OOD_token/Java250/model.bin \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --eval_data_file="../dataset/CodeNet_Java250/OOD_token/valid.jsonl" \
#    --test_data_file="../dataset/CodeNet_Java250/OOD_token/test.jsonl" \
#    --lang=java \
#    --num_labels 250 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --seed 123456
#
#CUDA_VISIBLE_DEVICES=1 python3 train_ts.py \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --model_path=./models/OOD_CST/Java250/model.bin \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --eval_data_file="../dataset/CodeNet_Java250/OOD_CST/valid.jsonl" \
#    --test_data_file="../dataset/CodeNet_Java250/OOD_CST/test.jsonl" \
#    --lang=java \
#    --num_labels 250 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --seed 123456
#
#
#CUDA_VISIBLE_DEVICES=1 python3 train_ts.py \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --model_path=./models/OOD/Java250/model.bin \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --eval_data_file="../dataset/CodeNet_Java250/OOD/valid.jsonl" \
#    --test_data_file="../dataset/CodeNet_Java250/OOD/test.jsonl" \
#    --lang=java \
#    --num_labels 250 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --seed 123456


CUDA_VISIBLE_DEVICES=2 python3 train_ts.py \
    --model_name_or_path=microsoft/graphcodebert-base \
    --model_path=./models/OOD_token/Python800/model.bin \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --eval_data_file="../dataset/CodeNet_Python800/OOD_token/valid.jsonl" \
    --test_data_file="../dataset/CodeNet_Python800/OOD_token/test.jsonl" \
    --lang=python \
    --num_labels 800 \
    --code_length 384 \
    --data_flow_length 128 \
    --seed 123456

CUDA_VISIBLE_DEVICES=1 python3 train_ts.py \
    --model_name_or_path=microsoft/graphcodebert-base \
    --model_path=./models/OOD_CST/Python800/model.bin \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --eval_data_file="../dataset/CodeNet_Python800/OOD_CST/valid.jsonl" \
    --test_data_file="../dataset/CodeNet_Python800/OOD_CST/test.jsonl" \
    --lang=python \
    --num_labels 800 \
    --code_length 384 \
    --data_flow_length 128 \
    --seed 123456

CUDA_VISIBLE_DEVICES=1 python3 train_ts.py \
    --model_name_or_path=microsoft/graphcodebert-base \
    --model_path=./models/OOD/Python800/model.bin \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --eval_data_file="../dataset/CodeNet_Python800/OOD/valid.jsonl" \
    --test_data_file="../dataset/CodeNet_Python800/OOD/test.jsonl" \
    --lang=python \
    --num_labels 800 \
    --code_length 384 \
    --data_flow_length 128 \
    --seed 123456
