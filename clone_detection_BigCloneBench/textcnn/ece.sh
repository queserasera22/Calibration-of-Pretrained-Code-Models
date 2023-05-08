CUDA_VISIBLE_DEVICES=0 python ece.py \
  --model_path="./models/test/model.pt" \
  --data_path='./data/corpus.pt' \
  --save_name="Clone-TextCNN"

CUDA_VISIBLE_DEVICES=3 python ece.py \
  --model_path="./models/ls_03/model.pt" \
  --data_path='./data/corpus.pt' \
  --save_name="Clone-TextCNN-LS-03"
