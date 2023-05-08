CUDA_VISIBLE_DEVICES=0 python ece.py \
  --model_path="./models/test_1234/model.pt" \
  --data_path='./data/corpus.pt' \
  --save_name="Exception-TextCNN"


#CUDA_VISIBLE_DEVICES=0 python ece.py \
#  --model_path="./models/ls/model.pt" \
#  --data_path='./data/corpus.pt' \
#  --save_name="Exception-TextCNN-LS"

CUDA_VISIBLE_DEVICES=1 python ece.py \
  --model_path="./models/ls_03/model.pt" \
  --data_path='./data/corpus.pt' \
  --save_name="Exception-TextCNN-LS-03"

