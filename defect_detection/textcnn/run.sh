#python corpus.py \
#	 --file-path '../dataset' \
#	 --save-data './data/corpus.pt'\
#	 --max-lenth 512\


CUDA_VISIBLE_DEVICES=2 python main.py \
  --batch-size 128 \
  --epochs 5 \
  --lr 1e-3 \
  --data './data/corpus.pt'  \
  --save_path './models/test1' \
  --cuda-able \
  --seed 123456
#
##--lr 1e-3 \  'eval_acc': 0.6328696925329429
##--lr 5e-4 \  'eval_acc': 0.6259150805270863
#
#CUDA_VISIBLE_DEVICES=0 python evaluate.py \
#  --model_path './models/model.pt' \
#  --data_path './data/corpus.pt'


###### label smoothing ######
CUDA_VISIBLE_DEVICES=1 python main.py \
  --batch-size 128 \
  --epochs 5 \
  --lr 1e-3 \
  --data './data/corpus.pt'  \
  --save_path './models/ls_03' \
  --cuda-able \
  --seed 123456

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
  --model_path './models/ls_03/model.pt' \
  --data_path './data/corpus.pt'