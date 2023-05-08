#CUDA_VISIBLE_DEVICES=2 python3 ece.py \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --model_path=./models/POJ104/model.bin \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --test_data_file=../dataset/POJ104/test.jsonl \
#    --num_labels 104 \
#    --lang=c \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --eval_batch_size 64 \
#    --save_name="POJ104-GraphCodeBERT" \
#    --seed 123456
#
#CUDA_VISIBLE_DEVICES=2 python3 ece.py \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --model_path=./models/Java250/model.bin \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --test_data_file=../dataset/CodeNet_Java250/test.jsonl \
#    --num_labels 250 \
#    --lang=java \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --eval_batch_size 64 \
#    --save_name="Java250-GraphCodeBERT" \
#    --seed 123456
#
#CUDA_VISIBLE_DEVICES=3 python3 ece.py \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --model_path=./models/Python800/model.bin \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --test_data_file=../dataset/CodeNet_Python800/test.jsonl \
#    --num_labels 800 \
#    --lang=python \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --eval_batch_size 64 \
#    --save_name="Python800-GraphCodeBERT" \
#    --seed 123456


CUDA_VISIBLE_DEVICES=2 python3 ece.py \
    --model_name_or_path=microsoft/graphcodebert-base \
    --model_path=./models/OOD_CST/ls_005/Java250/model.bin \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --test_data_file=../dataset/CodeNet_Java250/OOD_CST/test.jsonl \
    --num_labels 250 \
    --lang=java \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 64 \
    --save_name="Java250-GraphCodeBERT-OOD-CST-LS-005" \
    --seed 123456

#CUDA_VISIBLE_DEVICES=0 python3 ece.py \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --model_path=./models/OOD_CST/ls_01/Python800/model.bin \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --test_data_file=../dataset/CodeNet_Python800/OOD_CST/test.jsonl \
#    --num_labels 800 \
#    --lang=python \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --eval_batch_size 64 \
#    --save_name="Python800-GraphCodeBERT-OOD-CST-LS-01" \
#    --seed 123456
#
#CUDA_VISIBLE_DEVICES=2 python3 ece.py \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --model_path=./models/OOD/ls_01/Python800/model.bin \
#    --config_name=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --test_data_file=../dataset/CodeNet_Python800/OOD/test.jsonl \
#    --num_labels 800 \
#    --lang=python \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --eval_batch_size 64 \
#    --save_name="Python800-GraphCodeBERT-OOD-LS-01" \
#    --seed 123456