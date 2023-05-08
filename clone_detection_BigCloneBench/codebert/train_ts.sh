CUDA_VISIBLE_DEVICES=1 python train_ts.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert/checkpoint-best-f1/model.bin" \
    --eval_data_file="../dataset/valid_sampled.txt" \
    --test_data_file="../dataset/test_sampled.txt" \
    --eval_batch_size=16 \
    --block_size=512

CUDA_VISIBLE_DEVICES=2 python train_ts.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta/checkpoint-best-f1/model.bin" \
    --eval_data_file="../dataset/valid_sampled.txt" \
    --test_data_file="../dataset/test_sampled.txt" \
    --eval_batch_size=16 \
    --block_size=512