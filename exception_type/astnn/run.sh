##  generate preprocessed data
#python pipeline.py --lang python --data_path="../dataset/" --out_path="./data/"

# train
CUDA_VISIBLE_DEVICES=2 python main.py \
              --data_path ./data/  \
              --model_path ./models/test/ \
              --num_labels 20 \
              --train_batch_size 64 \
              --eval_batch_size 32 \
              --do_train \
              --do_test \
              --epochs 20 \
              --lr 0.002 \
              --seed 12345

### test
#CUDA_VISIBLE_DEVICES=2 python main.py \
#              --data_path ./data/  \
#              --model_path ./models \
#              --num_labels 20 \
#              --train_batch_size 64 \
#              --eval_batch_size 32 \
#              --do_test \
#              --epochs 20 \
#              --lr 0.002 \
#              --seed 123456

## train
#CUDA_VISIBLE_DEVICES=2 python main.py \
#              --data_path ./data/  \
#              --model_path ./models/ls/ \
#              --num_labels 20 \
#              --train_batch_size 64 \
#              --eval_batch_size 32 \
#              --do_train \
#              --do_test \
#              --epochs 20 \
#              --lr 0.002 \
#              --seed 123456


## train
#CUDA_VISIBLE_DEVICES=0 python main.py \
#              --data_path ./data/  \
#              --model_path ./models/ls_03/ \
#              --num_labels 20 \
#              --train_batch_size 64 \
#              --eval_batch_size 32 \
#              --do_train \
#              --do_test \
#              --epochs 20 \
#              --lr 0.002 \
#              --seed 123456