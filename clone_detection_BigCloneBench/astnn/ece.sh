CUDA_VISIBLE_DEVICES=3 python ece.py \
              --data_path ./data/  \
              --model_path ./models/best_f1/model.bin \
              --num_labels 2 \
              --eval_batch_size 16 \
              --save_name="Clone-ASTNN"