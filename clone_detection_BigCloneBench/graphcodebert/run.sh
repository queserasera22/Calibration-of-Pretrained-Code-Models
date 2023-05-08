#mkdir "logs"
#
#CUDA_VISIBLE_DEVICES=1 python3 main.py \
#    --output_dir=./models \
#    --config_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --do_train \
#    --train_data_file=../dataset/train_sampled.txt \
#    --eval_data_file=../dataset/valid_sampled.txt \
#    --test_data_file=../dataset/test_sampled.txt \
#    --epoch 4 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 16 \
#    --eval_batch_size 32 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train.log
#
#CUDA_VISIBLE_DEVICES=1 python3 main.py \
#    --output_dir=./models \
#    --config_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --do_test \
#    --train_data_file=../dataset/train_sampled.txt \
#    --eval_data_file=../dataset/test_sampled.txt \
#    --test_data_file=../dataset/test_sampled.txt \
#    --epoch 4 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 16 \
#    --eval_batch_size 32 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/eval.log


#CUDA_VISIBLE_DEVICES=1 python3 main.py \
#    --output_dir=./models/ls \
#    --config_name=microsoft/graphcodebert-base \
#    --model_name_or_path=microsoft/graphcodebert-base \
#    --tokenizer_name=microsoft/graphcodebert-base \
#    --do_train \
#    --train_data_file=../dataset/train_sampled.txt \
#    --eval_data_file=../dataset/valid_sampled.txt \
#    --test_data_file=../dataset/test_sampled.txt \
#    --epoch 4 \
#    --code_length 384 \
#    --data_flow_length 128 \
#    --train_batch_size 16 \
#    --eval_batch_size 32 \
#    --learning_rate 2e-5 \
#    --label_smoothing 0.1 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456 2>&1| tee ./logs/train_ls.log

CUDA_VISIBLE_DEVICES=3 python3 main.py \
    --output_dir=./models/ls_03 \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --do_test \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --epoch 4 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --label_smoothing 0.3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_ls_03.log
