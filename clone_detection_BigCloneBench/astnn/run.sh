## generate preprocessed data
#python pipeline.py --lang java --data_path="../dataset/" --out_path="./data/" --max_len 512
#
## train
#CUDA_VISIBLE_DEVICES=3 python main.py \
#              --data_path ./data/  \
#              --model_path ./models/best_f1 \
#              --num_labels 2 \
#              --train_batch_size 64 \
#              --eval_batch_size 64 \
#              --do_train \
#              --do_test \
#              --epochs 5 \
#              --lr 0.002 \
#              --seed 123456


## test
#CUDA_VISIBLE_DEVICES=0 python main.py \
#              --data_path ./data/  \
#              --model_path ./models/ \
#              --num_labels 2 \
#              --train_batch_size 64 \
#              --eval_batch_size 64 \
#              --do_test \
#              --epochs 5 \
#              --lr 0.002 \
#              --seed 123456



##### train with label smoothing #####
CUDA_VISIBLE_DEVICES=3 python main.py \
              --data_path ./data/  \
              --model_path ./models/ls_03 \
              --num_labels 2 \
              --train_batch_size 64 \
              --eval_batch_size 64 \
              --do_train \
              --do_test \
              --epochs 5 \
              --lr 0.002 \
              --seed 123456