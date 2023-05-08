CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert/test1/checkpoint-best-acc/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Defect-CodeBERT_test1"

CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta/test1/checkpoint-best-acc/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Defect-CodeBERTa_test1"


CUDA_VISIBLE_DEVICES=1 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert/ls_03/checkpoint-best-acc/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Defect-CodeBERT-LS-03"

CUDA_VISIBLE_DEVICES=1 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta/ls_03/checkpoint-best-acc/model.bin" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Defect-CodeBERTa-Ls-03"