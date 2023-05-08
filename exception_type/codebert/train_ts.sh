CUDA_VISIBLE_DEVICES=2 python train_ts.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert/model.bin" \
    --eval_data_file="../dataset/valid.jsonl" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512

CUDA_VISIBLE_DEVICES=0 python train_ts.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta/model.bin" \
    --eval_data_file="../dataset/valid.jsonl" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512

CUDA_VISIBLE_DEVICES=2 python train_ts.py \
    --model_name="microsoft/graphcodebert-base" \
    --model_path="models/graphcodebert_1234/model.bin" \
    --eval_data_file="../dataset/valid.jsonl" \
    --test_data_file="../dataset/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512