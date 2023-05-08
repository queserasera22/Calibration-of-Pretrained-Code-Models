## generate preprocessed data
#python pipeline.py --lang c --data_path="../dataset/POJ104" --out_path="./data/POJ104/"

## train
CUDA_VISIBLE_DEVICES=2 python main.py \
              --data_path ./data/POJ104/  \
              --model_path ./models/POJ104_12345 \
              --num_labels 104 \
              --train_batch_size 128 \
              --eval_batch_size 64 \
              --do_train \
              --do_test \
              --epochs 15 \
              --lr 0.002 \
              --seed 12345


## generate preprocessed data
#python pipeline.py --lang java --data_path="../dataset/CodeNet_Java250" --out_path="./data/Java250/"
##
###  train
CUDA_VISIBLE_DEVICES=0 python main.py \
              --data_path ./data/Java250/  \
              --model_path ./models/Java250_12345/ \
              --num_labels 250 \
              --train_batch_size 128 \
              --eval_batch_size 64 \
              --do_train \
              --do_test \
              --epochs 15 \
              --lr 0.002 \
              --seed 12345

# generate preprocessed data
#python pipeline.py --lang python --data_path="../dataset/CodeNet_Python800" --out_path="./data/Python800/"

###  train
CUDA_VISIBLE_DEVICES=2 python main.py \
              --data_path ./data/Python800/  \
              --model_path ./models/Python800_12345/ \
              --num_labels 800 \
              --train_batch_size 128 \
              --eval_batch_size 64 \
              --do_train \
              --do_test \
              --epochs 15 \
              --lr 0.002 \
              --seed 12342


######### train on OOD_CST dataset ########
## generate preprocessed data
#python pipeline.py --lang java --data_path="../dataset/CodeNet_Java250/OOD_CST" --out_path="./data/Java250/OOD_CST"

##  train
#CUDA_VISIBLE_DEVICES=2 python main.py \
#              --data_path ./data/Java250/OOD_CST  \
#              --model_path ./models/Java250/OOD_CST_test \
#              --num_labels 250 \
#              --train_batch_size 128 \
#              --eval_batch_size 64 \
#              --do_train \
#              --do_test \
#              --epochs 15 \
#              --lr 0.002 \
#              --seed 12345
#
#CUDA_VISIBLE_DEVICES=1 python main.py \
#              --data_path ./data/Java250/OOD_token  \
#              --model_path ./models/Java250/OOD_token/ls_005 \
#              --num_labels 250 \
#              --train_batch_size 128 \
#              --eval_batch_size 64 \
#              --do_train \
#              --do_test \
#              --epochs 15 \
#              --lr 0.002 \
#              --seed 12345

#
#
##python pipeline.py --lang java --data_path="../dataset/CodeNet_Java250/OOD_token" --out_path="./data/Java250/OOD_token"
##
#CUDA_VISIBLE_DEVICES=3 python main.py \
#              --data_path ./data/Java250/OOD  \
#              --model_path ./models/Java250/OOD_test \
#              --num_labels 250 \
#              --train_batch_size 128 \
#              --eval_batch_size 64 \
#              --do_train \
#              --do_test \
#              --epochs 15 \
#              --lr 0.002 \
#              --seed 1234


########### train on OOD_CST dataset ########
### generate preprocessed data
#python pipeline.py --lang python --data_path="../dataset/CodeNet_Python800/OOD_CST" --out_path="./data/Python800/OOD_CST"

##  train
#CUDA_VISIBLE_DEVICES=2 python main.py \
#              --data_path ./data/Python800/OOD_CST  \
#              --model_path ./models/Python800/OOD_CST/ls_01 \
#              --num_labels 800 \
#              --train_batch_size 128 \
#              --eval_batch_size 64 \
#              --do_train \
#              --do_test \
#              --epochs 20 \
#              --lr 0.002 \
#              --seed 12345

#python pipeline.py --lang python --data_path="../dataset/CodeNet_Python800/OOD_token" --out_path="./data/Python800/OOD_token"
#
#CUDA_VISIBLE_DEVICES=3 python main.py \
#              --data_path ./data/Python800/OOD_token  \
#              --model_path ./models/Python800/OOD_token_test/\
#              --num_labels 800 \
#              --train_batch_size 128 \
#              --eval_batch_size 64 \
#              --do_train \
#              --do_test \
#              --epochs 20 \
#              --lr 0.002 \
#              --seed 12345

#python pipeline.py --lang python --data_path="../dataset/CodeNet_Python800/OOD" --out_path="./data/Python800/OOD"
#
#CUDA_VISIBLE_DEVICES=1 python main.py \
#              --data_path ./data/Python800/OOD  \
#              --model_path ./models/Python800/OOD/ls_01 \
#              --num_labels 800 \
#              --train_batch_size 64 \
#              --eval_batch_size 64 \
#              --do_train \
#              --do_test \
#              --epochs 20 \
#              --lr 0.002 \
#              --seed 123456