import argparse
import pickle
import sys
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
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel


sys.path.append("../../")
from ece_utils import draw_reliability_graph

set_seed(123456)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer):
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)

    model.eval()

    logits = []
    labels = []
    y_preds = []

    start_time = time.time()  # 程序开始时间
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            input_id = batch[0].to(device)
            test_label = batch[1].to(device)

            output = model(input_id)

            logits.append(output.cpu())
            labels.append(test_label.cpu())
            y_preds.append(torch.argmax(output, dim=1).cpu())

    end_time = time.time()  # 程序结束时间

    print('Test time: %.3fms' % ((end_time - start_time) / len(eval_dataloader) * 1000))

    logits = torch.cat(logits, 0)
    labels = torch.cat(labels, 0)
    y_preds = torch.cat(y_preds, 0)

    model_dir = "/".join(args.model_path.split("/")[:-1])
    out_data = {'logits': logits, 'labels': labels, 'preds': y_preds}
    with open(os.path.join(model_dir, "test_pred.pkl"), 'wb') as f:
        pickle.dump(out_data, f)


    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
    draw_reliability_graph(logits, labels, 10, save_path, args.save_name)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/codebert-base", type=str, required=True,
                        help="The model name or path.")
    parser.add_argument("--model_path", default="models/codebert_POJ104/model.pt", type=str,
                        help="The path to the fine-tuned model")
    parser.add_argument("--test_data_file", default="../dataset/test.jsonl", type=str, required=True,
                        help="The input testing data file (a json file).")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--save_name", type=str,
                        help="The name of saved figures.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)

    model_dir = "/".join(args.model_path.split("/")[:-1])
    if os.path.exists(os.path.join(model_dir, "test_pred.pkl")):
        print("Load test_pred.pkl")
        with open(os.path.join(model_dir, "test_pred.pkl"), 'rb') as f:
            loaded_tensors = pickle.load(f)
        labels = loaded_tensors['labels']
        logits = loaded_tensors['logits']

        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
        draw_reliability_graph(logits, labels, 10, save_path, args.save_name)
        return

    config = RobertaConfig.from_pretrained(args.model_name)
    config.num_labels = 2

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.do_lower_case = True

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = Model(RobertaModel.from_pretrained(
        args.model_name, config=config), config, args)

    model.load_state_dict(torch.load(args.model_path), strict=False)

    evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
