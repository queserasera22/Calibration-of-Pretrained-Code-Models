CUDA_VISIBLE_DEVICES=0 python3 ece.py \
    --model_name_or_path=microsoft/graphcodebert-base \
    --model_path=./models/model.bin \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --test_data_file="../dataset/test.jsonl" \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 64 \
    --save_name="Defect-GraphCodeBERT" \
    --seed 123456


CUDA_VISIBLE_DEVICES=1 python3 ece.py \
    --model_name_or_path=microsoft/graphcodebert-base \
    --model_path=./models/ls_01/model.bin \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --test_data_file="../dataset/test.jsonl" \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 64 \
    --save_name="Defect-GraphCodeBERT-LS-01" \
    --seed 123456