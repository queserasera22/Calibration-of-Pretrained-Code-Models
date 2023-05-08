## generate preprocessed data
#python pipeline.py --lang c --data_path="../dataset/" --out_path="./data/" --max_len 512

## train
#CUDA_VISIBLE_DEVICES=0 python main.py \
#              --data_path ./data/  \
#              --model_path ./models/ \
#              --num_labels 2 \
#              --train_batch_size 64 \
#              --eval_batch_size 64 \
#              --do_train \
#              --do_test \
#              --epochs 20 \
#              --lr 0.002 \
#              --seed 123456


######### train using label_smoothing
CUDA_VISIBLE_DEVICES=0 python main.py \
              --data_path ./data/  \
              --model_path ./models/test_12345/ \
              --num_labels 2 \
              --train_batch_size 64 \
              --eval_batch_size 64 \
              --do_train \
              --do_test \
              --epochs 20 \
              --lr 0.002 \
              --seed 12345