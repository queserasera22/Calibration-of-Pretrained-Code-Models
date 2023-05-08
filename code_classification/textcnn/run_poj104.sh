# preprocess the dataset
python corpus.py \
  --file-path '../dataset/POJ104' \
  --save-data './data/POJ104.pt' \
  --max-lenth 512

# train the textcnn model
CUDA_VISIBLE_DEVICES=0 python main.py \
  --batch-size 128 \
  --lr 0.001 \
  --epochs 20 \
  --save_path './models/CNN_POJ104_test' \
  --data './data/POJ104.pt' \
  --cuda-able \
  --seed 12345

# evaluate the textcnn model
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model_path './models/CNN_POJ104/model.pt' \
  --data './data/POJ104.pt'
