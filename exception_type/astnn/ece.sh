CUDA_VISIBLE_DEVICES=3 python ece.py \
              --data_path ./data/  \
              --model_path ./models/model.bin \
              --num_labels 20 \
              --eval_batch_size 16 \
              --save_name="Exception-ASTNN"

CUDA_VISIBLE_DEVICES=1 python ece.py \
              --data_path ./data/  \
              --model_path ./models/ls_12345/model.bin \
              --num_labels 20 \
              --eval_batch_size 16 \
              --save_name="Exception-ASTNN-LS-test"