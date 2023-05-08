mkdir logs

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train \
    --model_name=huggingface/CodeBERTa-small-v1 \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --output_dir=./models/codeberta/test_123 \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123 2>&1| tee ./logs/train_codeberta_test123.log

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_eval \
    --model_name=huggingface/CodeBERTa-small-v1 \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/test.jsonl \
    --output_dir=./models/codeberta/test_123 \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123 2>&1| tee ./logs/eval_codeberta_test123.log