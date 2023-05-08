import argparse
import pickle
import time
import os
import warnings
import logging
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch

from models import CloneModel
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

import multiprocessing
from configs import set_seed
from utils import get_filenames, get_elapse_time, load_and_cache_clone_data
from models import get_model_size

import sys
sys.path.append("../../")
from temperature_scaling import ModelWithTemperature

cpu_cont = multiprocessing.cpu_count()
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)



def evaluate(args, model, eval_data, test_data):

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(model)

    # Tune the model temperature, and save the results
    model.set_temperature(eval_dataloader)

    save_path = os.path.join(os.path.dirname(args.load_model_path), 'ts')
    os.makedirs(save_path, exist_ok=True)
    model_filename = os.path.join(save_path, 'model.pt')
    torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model saved to %s' % model_filename)


    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    model.evaluate(test_data_loader, save_path)

    print('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=['summarize', 'concode', 'translate', 'refine', 'defect', 'clone', 'multi_task', 'classification'])
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--eval_filename", default="../dataset/valid.jsonl", type=str, required=True,
                        help="The input valid data file (a json file).")
    parser.add_argument("--test_filename", default="../dataset/test.jsonl", type=str, required=True,
                        help="The input testing data file (a json file).")
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--data_num", default=-1, type=int)

    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--save_name", type=str,
                        help="The name of saved figures.")
    parser.add_argument("--max_source_length", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="The number of labels to be classified.")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    # model.resize_token_embeddings(32000)

    model = CloneModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(args.device)

    pool = multiprocessing.Pool(cpu_cont)
    eval_examples, eval_data = load_and_cache_clone_data(args, args.eval_filename, pool, tokenizer, 'valid',
                                                         False)
    test_examples, test_data = load_and_cache_clone_data(args, args.test_filename, pool, tokenizer, 'test',
                                                         False)

    evaluate(args, model, eval_data, test_data)


if __name__ == "__main__":
    main()
