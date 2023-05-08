CUDA_VISIBLE_DEVICES=1 python train_ts.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert_OOD_CST/Java250/model.bin" \
    --eval_data_file="../dataset/CodeNet_Java250/OOD_CST/valid.jsonl" \
    --test_data_file="../dataset/CodeNet_Java250/OOD_CST/test.jsonl" \
    --eval_batch_size=32 \
    --num_labels 250 \
    --block_size=512
#
CUDA_VISIBLE_DEVICES=1 python train_ts.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta_OOD_CST/Java250/model.bin" \
    --eval_data_file="../dataset/CodeNet_Java250/OOD_CST/valid.jsonl" \
    --test_data_file="../dataset/CodeNet_Java250/OOD_CST/test.jsonl" \
    --eval_batch_size=32 \
    --num_labels 250 \
    --block_size=512

CUDA_VISIBLE_DEVICES=2 python train_ts.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert_OOD/Java250/model.bin" \
    --eval_data_file="../dataset/CodeNet_Java250/OOD/valid.jsonl" \
    --test_data_file="../dataset/CodeNet_Java250/OOD/test.jsonl" \
    --eval_batch_size=32 \
    --num_labels 250 \
    --block_size=512
#
CUDA_VISIBLE_DEVICES=3 python train_ts.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta_OOD/Java250/model.bin" \
    --eval_data_file="../dataset/CodeNet_Java250/OOD/valid.jsonl" \
    --test_data_file="../dataset/CodeNet_Java250/OOD/test.jsonl" \
    --eval_batch_size=32 \
    --num_labels 250 \
    --block_size=512

CUDA_VISIBLE_DEVICES=1 python train_ts.py \
    --model_name="microsoft/graphcodebert-base" \
    --model_path="models/graphcodebert_OOD/Java250/model.bin" \
    --eval_data_file="../dataset/CodeNet_Java250/OOD/valid.jsonl" \
    --test_data_file="../dataset/CodeNet_Java250/OOD/test.jsonl" \
    --eval_batch_size=32 \
    --num_labels 250 \
    --block_size=512

CUDA_VISIBLE_DEVICES=2 python train_ts.py \
    --model_name="microsoft/graphcodebert-base" \
    --model_path="models/graphcodebert_OOD/Python800/model.bin" \
    --eval_data_file="../dataset/CodeNet_Python800/OOD/valid.jsonl" \
    --test_data_file="../dataset/CodeNet_Python800/OOD/test.jsonl" \
    --eval_batch_size=32 \
    --num_labels 800 \
    --block_size=512