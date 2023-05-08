CUDA_VISIBLE_DEVICES=0 python ece.py \
              --data_path ./data/POJ104  \
              --model_path ./models/POJ104/model.bin \
              --num_labels 104 \
              --eval_batch_size 64 \
              --save_name="POJ104-ASTNN"

CUDA_VISIBLE_DEVICES=0 python ece.py \
              --data_path ./data/Java250  \
              --model_path ./models/Java250/model.bin \
              --num_labels 250 \
              --eval_batch_size 64 \
              --save_name="Java250-ASTNN"

CUDA_VISIBLE_DEVICES=3 python ece.py \
              --data_path ./data/Python800  \
              --model_path ./models/Python800/model.bin \
              --num_labels 800 \
              --eval_batch_size 64 \
              --save_name="Python800-ASTNN"


CUDA_VISIBLE_DEVICES=0 python ece.py \
              --data_path ./data/Java250/OOD_CST  \
              --model_path ./models/Java250/OOD_CST/model.bin \
              --num_labels 250 \
              --eval_batch_size 64 \
              --save_name="Java250-ASTNN-OOD-CST"

CUDA_VISIBLE_DEVICES=0 python ece.py \
              --data_path ./data/Java250/OOD  \
              --model_path ./models/Java250/OOD/model.bin \
              --num_labels 250 \
              --eval_batch_size 64 \
              --save_name="Java250-ASTNN-OOD"

CUDA_VISIBLE_DEVICES=0 python ece.py \
              --data_path ./data/Python800/OOD  \
              --model_path ./models/Python800/OOD/ls_01/model.bin \
              --num_labels 800 \
              --eval_batch_size 64 \
              --save_name="Python800-ASTNN-OOD-LS-01"


CUDA_VISIBLE_DEVICES=0 python ece.py \
              --data_path ./data/Python800/OOD_token  \
              --model_path ./models/Python800/OOD_token_test/model.bin \
              --num_labels 800 \
              --eval_batch_size 64 \
              --save_name="Python800-ASTNN-OOD-token-test"