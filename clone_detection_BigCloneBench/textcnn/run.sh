#python corpus.py \
#	 --file-path '../dataset' \
#	 --save-data './data/corpus.pt'\
#	 --max-lenth 512\


#CUDA_VISIBLE_DEVICES=0 python main.py \
#  --batch-size 128 \
#  --epochs 10 \
#  --lr 1e-3 \
#  --data './data/corpus.pt'  \
#  --save_path './models/test1' \
#  --cuda-able \
#  --seed 1234
#
##--lr 5e-4 \
#
#CUDA_VISIBLE_DEVICES=0 python evaluate.py \
#  --model_path './models/model.pt' \
#  --data_path './data/corpus.pt'


######### label_smoothing #########
CUDA_VISIBLE_DEVICES=3 python main.py \
  --batch-size 128 \
  --epochs 10 \
  --lr 1e-3 \
  --data './data/corpus.pt'  \
  --save_path './models/ls_03' \
  --cuda-able \
  --seed 123456

#--lr 5e-4 \

CUDA_VISIBLE_DEVICES=3 python evaluate.py \
  --model_path './models/ls_03/model.pt' \
  --data_path './data/corpus.pt'
