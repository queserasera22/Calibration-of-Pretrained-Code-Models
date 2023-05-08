CUDA_VISIBLE_DEVICES=2 python ece.py \
  --model_path="./models/test/model.pt" \
  --data_path='./data/corpus.pt' \
  --save_name="Defect-TextCNN-test"

CUDA_VISIBLE_DEVICES=1 python ece.py \
  --model_path="./models/ls_03/model.pt" \
  --data_path='./data/corpus.pt' \
  --save_name="Defect-TextCNN-LS-03"