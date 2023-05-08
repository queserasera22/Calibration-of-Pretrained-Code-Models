import pickle

import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import time
import os
import warnings
import logging
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch

from model import Model
from utils import set_seed, TextDataset
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

import sys
sys.path.append("../../")
from temperature_scaling import ModelWithTemperature

set_seed(123456)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train_temperature(args, model, tokenizer):
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(model)

    # Tune the model temperature, and save the results
    model.set_temperature(eval_dataloader)

    save_path = os.path.join(os.path.dirname(args.model_path), 'ts')
    os.makedirs(save_path, exist_ok=True)
    model_filename = os.path.join(save_path, 'model.pt')
    torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model saved to %s' % model_filename)

    test_dataset = TextDataset(tokenizer, args, args.test_data_file)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)

    model.evaluate(test_dataloader, save_path)

    print('Done!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/codebert-base", type=str, required=True,
                        help="The model name or path.")
    parser.add_argument("--model_path", default="models/codebert_POJ104/model.pt", type=str,
                        help="The path to the fine-tuned model")
    parser.add_argument("--eval_data_file", default="../dataset/valid.jsonl", type=str, required=True,
                        help="The input testing data file (a json file).")
    parser.add_argument("--test_data_file", default="../dataset/test.jsonl", type=str, required=True,
                        help="The input testing data file (a json file).")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)

    config = RobertaConfig.from_pretrained(args.model_name)
    config.num_labels = 2

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.do_lower_case = True

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = Model(RobertaForSequenceClassification.from_pretrained(args.model_name, config=config))

    model.load_state_dict(torch.load(args.model_path), strict=False)

    train_temperature(args, model, tokenizer)


if __name__ == "__main__":
    main()
