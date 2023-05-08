CUDA_VISIBLE_DEVICES=0 python ece.py \
              --data_path ./data/  \
              --model_path ./models/model.bin \
              --num_labels 2 \
              --eval_batch_size 64 \
              --save_name="Defect-ASTNN"

CUDA_VISIBLE_DEVICES=1 python ece.py \
              --data_path ./data/  \
              --model_path ./models/test_12345/model.bin \
              --num_labels 2 \
              --eval_batch_size 64 \
              --save_name="Defect-ASTNN-test_12345"