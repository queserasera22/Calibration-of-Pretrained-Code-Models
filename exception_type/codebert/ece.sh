CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Exception-CodeBERT"
#
CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Exception-CodeBERTa"


CUDA_VISIBLE_DEVICES=1 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert_ls_02/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Exception-CodeBERT-LS-02"

CUDA_VISIBLE_DEVICES=1 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta_ls_02/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Exception-CodeBERTa-LS-02"


CUDA_VISIBLE_DEVICES=1 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert_ls_03/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Exception-CodeBERT-LS-03"

CUDA_VISIBLE_DEVICES=1 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta_ls_03/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Exception-CodeBERTa-LS-03"

CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name="microsoft/graphcodebert-base" \
    --model_path="models/graphcodebert_1234/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Exception-GraphCodeBERT"