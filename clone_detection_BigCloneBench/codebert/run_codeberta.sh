#mkdir logs
#
#CUDA_VISIBLE_DEVICES=1 python main.py \
#    --do_train \
#    --model_name=huggingface/CodeBERTa-small-v1 \
#    --train_data_file=../dataset/train_sampled.txt \
#    --eval_data_file=../dataset/valid_sampled.txt \
#    --output_dir=./models/codeberta/ \
#    --epoch 5 \
#    --block_size 512 \
#    --train_batch_size 24 \
#    --eval_batch_size 64 \
#    --learning_rate 5e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_codeberta.log
#
#CUDA_VISIBLE_DEVICES=1 python main.py \
#    --do_eval \
#    --model_name=huggingface/CodeBERTa-small-v1 \
#    --train_data_file=../dataset/train_sampled.txt \
#    --eval_data_file=../dataset/test_sampled.txt \
#    --output_dir=./models/codeberta/ \
#    --block_size 512 \
#    --eval_batch_size 64 \
#    --seed 123456 2>&1| tee ./logs/eval_codeberta.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --do_train \
    --model_name=huggingface/CodeBERTa-small-v1 \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --output_dir=./models/codeberta_ls/ \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --label_smoothing 0.1 \
    --seed 123456 2>&1| tee ./logs/train_codeberta_ls.log

CUDA_VISIBLE_DEVICES=2 python main.py \
    --do_eval \
    --model_name=huggingface/CodeBERTa-small-v1 \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled.txt \
    --output_dir=./models/codeberta_ls/ \
    --block_size 512 \
    --eval_batch_size 64 \
    --label_smoothing 0.1 \
    --seed 123456 2>&1| tee ./logs/eval_codeberta_ls.log