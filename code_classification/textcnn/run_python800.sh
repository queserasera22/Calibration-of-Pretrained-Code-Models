## preprocess the dataset
#python corpus.py \
#  --file-path '../dataset/CodeNet_Python800' \
#  --save-data './data/Python800.pt' \
#  --max-lenth 512
#
## train the textcnn model
CUDA_VISIBLE_DEVICES=3 python main.py \
  --batch-size 256 \
  --lr 0.001 \
  --epochs 20 \
  --save_path './models/CNN_Python800' \
  --data './data/Python800.pt' \
  --cuda-able \
  --seed 123456
#
## evaluate the textcnn model
#CUDA_VISIBLE_DEVICES=3 python evaluate.py \
#  --model_path './models/CNN_Python800/model.pt' \
#  --data './data/Python800.pt'


### preprocess the dataset
#python corpus.py \
#  --file-path '../dataset/CodeNet_Python800/OOD_token' \
#  --save-data './data/OOD_token/Python800.pt' \
#  --max-lenth 512

## train the textcnn model
CUDA_VISIBLE_DEVICES=2 python main.py \
  --batch-size 256 \
  --lr 0.001 \
  --epochs 20 \
  --save_path './models/CNN_Python800/OOD_token_test' \
  --data './data/OOD_token/Python800.pt' \
  --cuda-able \
  --seed 12345
#
## evaluate the textcnn model
#CUDA_VISIBLE_DEVICES=1 python evaluate.py \
#  --model_path './models/CNN_Python800/OOD_token/ls_01/model.pt' \
#  --data './data/OOD_token/Python800.pt'


## preprocess the dataset
python corpus.py \
  --file-path '../dataset/CodeNet_Python800/OOD' \
  --save-data './data/OOD/Python800.pt' \
  --max-lenth 512

# train the textcnn model
CUDA_VISIBLE_DEVICES=0 python main.py \
  --batch-size 256 \
  --lr 0.001 \
  --epochs 20 \
  --save_path './models/CNN_Python800/OOD/ls_01' \
  --data './data/OOD/Python800.pt' \
  --cuda-able \
  --seed 123456

# evaluate the textcnn model
CUDA_VISIBLE_DEVICES=1 python evaluate.py \
  --model_path './models/CNN_Python800/OOD/model.pt' \
  --data './data/OOD/Python800.pt'