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




## Acknowledgement

We are very grateful that the authors of CodeBERTa, CodeBERT, GraphCodeBERT, CodeT5, TextCNN and ASTNN make their models and code publicly available so that we can build this repository on top of their code.
