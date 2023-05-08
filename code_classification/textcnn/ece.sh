CUDA_VISIBLE_DEVICES=0 python ece.py \
  --model_path="./models/CNN_POJ104/model.pt" \
  --data_path='./data/POJ104.pt' \
  --save_name="POJ104-TextCNN"

CUDA_VISIBLE_DEVICES=0 python ece.py \
  --model_path="./models/CNN_Java250/model.pt" \
  --data_path='./data/Java250.pt' \
  --save_name="Java250-TextCNN"

CUDA_VISIBLE_DEVICES=0 python ece.py \
  --model_path="./models/CNN_Python800/model.pt" \
  --data_path='./data/Python800.pt' \
  --save_name="Python800-TextCNN"


CUDA_VISIBLE_DEVICES=2 python ece.py \
  --model_path="./models/CNN_Java250/OOD_CST/model.pt" \
  --data_path='./data/OOD_CST/Java250.pt' \
  --save_name="Java250-TextCNN-OOD-CST"


CUDA_VISIBLE_DEVICES=2 python ece.py \
  --model_path="./models/CNN_Java250/OOD/model.pt" \
  --data_path='./data/OOD/Java250.pt' \
  --save_name="Java250-TextCNN-OOD"


CUDA_VISIBLE_DEVICES=0 python ece.py \
  --model_path="./models/CNN_Python800/OOD_token_test/model.pt" \
  --data_path='./data/OOD_token/Python800.pt' \
  --save_name="Python800-TextCNN-OOD-token-test"
#
  CUDA_VISIBLE_DEVICES=0 python ece.py \
  --model_path="./models/CNN_Python800/OOD/ls_01/model.pt" \
  --data_path='./data/OOD/Python800.pt' \
  --save_name="Python800-TextCNN-OOD-LS-01"