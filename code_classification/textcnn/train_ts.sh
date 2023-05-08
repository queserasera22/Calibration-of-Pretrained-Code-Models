CUDA_VISIBLE_DEVICES=2 python train_ts.py \
              --data_path ./data/OOD_CST/Java250.pt  \
              --model_path ./models/CNN_Java250/OOD_CST/model.pt

CUDA_VISIBLE_DEVICES=2 python train_ts.py \
              --data_path ./data/OOD_token/Java250.pt  \
              --model_path ./models/CNN_Java250/OOD_token/model.pt

CUDA_VISIBLE_DEVICES=2 python train_ts.py \
              --data_path ./data/OOD/Java250.pt  \
              --model_path ./models/CNN_Java250/OOD/model.pt


CUDA_VISIBLE_DEVICES=0 python train_ts.py \
              --data_path ./data/OOD/Python800.pt  \
              --model_path ./models/CNN_Python800/OOD/model.pt

CUDA_VISIBLE_DEVICES=0 python train_ts.py \
              --data_path ./data/OOD_token/Python800.pt  \
              --model_path ./models/CNN_Python800/OOD_token/model.pt