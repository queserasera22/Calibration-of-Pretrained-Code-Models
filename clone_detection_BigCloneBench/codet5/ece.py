import argparse
import pickle

import time
import os
import warnings
import logging
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch

from models import CloneModel
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
from ece_utils import draw_reliability_graph

cpu_cont = multiprocessing.cpu_count()
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_examples, eval_data):
    model_dir = "/".join(args.load_model_path.split("/")[:-1])
    if os.path.exists(os.path.join(model_dir, "test_pred.pkl")):
        print("Load test_pred.pkl")
        with open(os.path.join(model_dir, "test_pred.pkl"), 'rb') as f:
            loaded_tensors = pickle.load(f)

        # to tensor
        y_trues = torch.tensor(loaded_tensors['labels'])
        logits = torch.tensor(loaded_tensors['logits'])
    else:
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation  *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        logits = []
        y_trues = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit = model(inputs, labels)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu())
                y_trues.append(labels.cpu())
            nb_eval_steps += 1

        logits = torch.cat(logits, 0)
        y_trues = torch.cat(y_trues, 0)
        y_preds = torch.argmax(logits, dim=1)

        eval_acc = accuracy_score(y_trues, y_preds)
        recall = recall_score(y_trues, y_preds)
        precision = precision_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds)
        result = {
            "eval_accuracy": float(eval_acc),
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1)
        }

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))
        logger.info("  " + "*" * 20)

        model_dir = "/".join(args.load_model_path.split("/")[:-1])
        out_data = {'logits': logits, 'labels': y_trues, 'preds': y_preds}
        with open(os.path.join(model_dir, "test_pred.pkl"), 'wb') as f:
            pickle.dump(out_data, f)


    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
    draw_reliability_graph(logits, y_trues, 10, save_path, args.save_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=['summarize', 'concode', 'translate', 'refine', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])

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
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    parser.add_argument("--label_smoothing", default=0.0, type=float,
                        help="The label smoothing epsilon to apply (zero means no label smoothing).")

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
    eval_examples, eval_data = load_and_cache_clone_data(args, args.test_filename, pool, tokenizer, 'test',
                                                         False)

    evaluate(args, model, eval_examples, eval_data)


if __name__ == "__main__":
    main()
