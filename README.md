# Calibration of Pre-trained Code models
This repository contains the code and data for the paper "On Calibration of Pre-trained Code models"


## Environment configuration
To reproduce our experiments, machines with GPUs and NVIDIA CUDA toolkit are required.

The environment dependencies are listed in the file "requirements.txt". You can create conda environment to install required dependencies:

```
conda create --name <env> --file requirements.txt
```

## Tasks and Datasets
We evaluate the calibraiton of pre-trained code models on different code understanding tasks. The datasets can be downloaded from the following sources:
* Code Classification: we use three datasets in our code classification experiments, namely [POJ104](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104), [Java250 and Python800](https://github.com/IBM/Project_CodeNet). Note that the partition of datasets follows [CodeNet paper](https://arxiv.org/abs/2105.12655). For each dataset, 20\% of the data is used as a testing set, while the rest is divided in 4:1 for training and validation.
* Clone Detection: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench
* Defect Detection: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection
* Exception Type: We directly use the dataset from this [paper](https://arxiv.org/abs/2302.04026)


## ECE Calculation
We employ Expected Calibration Error(ECE) to measure the calibration of pre-trained code models , and we use reliability diagrams to visualize the calibration the difference between the average predicted accuracy and the
confidence of the models.

The file ["ece_utils"](./ece_utils.py) provides the utils to calculate the ECE and draw the reliability diagrams.


## Calibaration Methods

We evaluate the effectiveness of two popular and simple calibration methods on pre-trained code models.

* Temperature Scaling
* Lable Smoothing

## How to Run
For each experiment presented in our paper, the code files and scripts are organized in separate subfolders. To facilitate ease of use, we have provided a "run.sh" file within each subfolder, which contains the necessary commands to train or evaluate the corresponding models. For instance, to fine-tune the CodeBERT model on the task of code clone detection, please run the following instructions:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train \
    --model_name=microsoft/codebert-base \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --output_dir=./models/codebert/ \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 12 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./logs/train_codebert.log
```

And to evaluate the fine-tuned models, run:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_eval \
    --model_name=microsoft/codebert-base \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test.txt \
    --output_dir=./models/codebert/ \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 12 \
    --eval_batch_size 64 \
    --seed 123456 2>&1| tee ./logs/test_codebert.log
```

To calibrate the fine-tuned models with Temperature Scaling, following the instructions in "train_ts.sh" files:
```
CUDA_VISIBLE_DEVICES=1 python train_ts.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert/checkpoint-best-f1/model.bin" \
    --eval_data_file="../dataset/valid_sampled.txt" \
    --test_data_file="../dataset/test_sampled.txt" \
    --eval_batch_size=16 \
    --block_size=512
```




## Acknowledgement

We are very grateful that the authors of CodeBERTa, CodeBERT, GraphCodeBERT, CodeT5, TextCNN and ASTNN make their models and code publicly available so that we can build this repository on top of their code.
