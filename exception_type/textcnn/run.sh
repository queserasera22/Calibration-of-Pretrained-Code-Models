# preprocess the dataset
#python corpus.py \
#  --file-path '../dataset' \
#  --save-data './data/corpus.pt' \
#  --max-lenth 512

## train the textcnn model
CUDA_VISIBLE_DEVICES=0 python main.py \
  --batch-size 128 \
  --lr 0.001 \
  --epochs 10 \
  --save_path './models/test_1234' \
  --data './data/corpus.pt' \
  --cuda-able \
  --seed 1234
#
## evaluate the textcnn model
#CUDA_VISIBLE_DEVICES=0 python evaluate.py \
#  --model_path './models/model.pt' \
#  --data './data/corpus.pt'

# train the textcnn model with label smoothing
CUDA_VISIBLE_DEVICES=0 python main.py \
  --batch-size 128 \
  --lr 0.001 \
  --epochs 10 \
  --save_path './models/ls_03' \
  --data './data/corpus.pt' \
  --cuda-able \
  --seed 123456
#
## evaluate the textcnn model
#CUDA_VISIBLE_DEVICES=0 python evaluate.py \
#  --model_path './models/ls_03/model.pt' \
#  --data './data/corpus.pt'
