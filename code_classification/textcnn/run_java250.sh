## preprocess the dataset
#python corpus.py \
#  --file-path '../dataset/CodeNet_Java250' \
#  --save-data './data/Java250.pt' \
#  --max-lenth 512
#
## train the textcnn model
CUDA_VISIBLE_DEVICES=0 python main.py \
  --batch-size 256 \
  --lr 0.001 \
  --epochs 20 \
  --save_path './models/CNN_Java250' \
  --data './data/Java250.pt' \
  --cuda-able \
  --seed 123456
#
## evaluate the textcnn model
#CUDA_VISIBLE_DEVICES=0 python evaluate.py \
#  --model_path './models/CNN_Java250/model.pt' \
#  --data './data/Java250.pt'

######### train on OOD_CST_test dataset ########
##  preprocess the dataset
#python corpus.py \
#  --file-path '../dataset/CodeNet_Java250/OOD_CST' \
#  --save-data './data/OOD_CST/Java250.pt' \
#  --max-lenth 512

## train the textcnn model
CUDA_VISIBLE_DEVICES=0 python main.py \
  --batch-size 512 \
  --lr 0.001 \
  --epochs 20 \
  --save_path './models/CNN_Java250/OOD_CST/ls_005' \
  --data './data/OOD_CST/Java250.pt' \
  --cuda-able \
  --seed 123456
#
## evaluate the textcnn model
#CUDA_VISIBLE_DEVICES=2 python evaluate.py \
#  --model_path './models/CNN_Java250/OOD_CST_test/model.pt' \
#  --data './data/OOD_CST/Java250.pt'

######### train on OOD_token_test dataset ########
##  preprocess the dataset
#python corpus.py \
#  --file-path '../dataset/CodeNet_Java250/OOD_token' \
#  --save-data './data/OOD_token/Java250.pt' \
#  --max-lenth 512

# train the textcnn model
CUDA_VISIBLE_DEVICES=2 python main.py \
  --batch-size 512 \
  --lr 0.001 \
  --epochs 20 \
  --save_path './models/CNN_Java250/OOD_token/ls_005' \
  --data './data/OOD_token/Java250.pt' \
  --cuda-able \
  --seed 123456
#
## evaluate the textcnn model
#CUDA_VISIBLE_DEVICES=2 python evaluate.py \
#  --model_path './models/CNN_Java250/OOD_token_test/model.pt' \
#  --data './data/OOD_token/Java250.pt'


CUDA_VISIBLE_DEVICES=3 python main.py \
  --batch-size 512 \
  --lr 0.001 \
  --epochs 20 \
  --save_path './models/CNN_Java250/OOD/ls_005' \
  --data './data/OOD/Java250.pt' \
  --cuda-able \
  --seed 123456

# evaluate the textcnn model
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model_path './models/CNN_Java250/OOD_token/ls_005/model.pt' \
  --data './data/OOD_token/Java250.pt'
