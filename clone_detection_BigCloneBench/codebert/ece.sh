CUDA_VISIBLE_DEVICES=3 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert/checkpoint-best-f1/model.bin" \
    --test_data_file="../dataset/test_sampled.txt" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Clone-CodeBERT"

CUDA_VISIBLE_DEVICES=3 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta/checkpoint-best-f1/model.bin" \
    --test_data_file="../dataset/test_sampled.txt" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Clone-CodeBERTa"




CUDA_VISIBLE_DEVICES=3 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert_ls_03/checkpoint-best-f1/model.bin" \
    --test_data_file="../dataset/test_sampled.txt" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Clone-CodeBERT-LS-03"

CUDA_VISIBLE_DEVICES=3 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta_ls_03/checkpoint-best-f1/model.bin" \
    --test_data_file="../dataset/test_sampled.txt" \
    --eval_batch_size=64 \
    --block_size=512 \
    --save_name="Clone-CodeBERTa-LS-03"