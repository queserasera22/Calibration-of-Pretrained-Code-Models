CUDA_VISIBLE_DEVICES=0 python train_ts.py \
              --data_path ./data/Java250/OOD_CST  \
              --model_path ./models/Java250/OOD_CST/model.bin \
              --num_labels 250 \
#
#
CUDA_VISIBLE_DEVICES=1 python train_ts.py \
              --data_path ./data/Java250/OOD_token  \
              --model_path ./models/Java250/OOD_token/model.bin \
              --num_labels 250
#
#
CUDA_VISIBLE_DEVICES=2 python train_ts.py \
              --data_path ./data/Java250/OOD  \
              --model_path ./models/Java250/OOD/model.bin \
              --num_labels 250

CUDA_VISIBLE_DEVICES=2 python train_ts.py \
              --data_path ./data/Python800/OOD_CST  \
              --model_path ./models/Python800/OOD_CST/model.bin \
              --num_labels 800


CUDA_VISIBLE_DEVICES=0 python train_ts.py \
              --data_path ./data/Python800/OOD_token  \
              --model_path ./models/Python800/OOD_token_test/model.bin \
              --num_labels 800


CUDA_VISIBLE_DEVICES=1 python train_ts.py \
              --data_path ./data/Python800/OOD  \
              --model_path ./models/Python800/OOD/model.bin \
              --num_labels 800