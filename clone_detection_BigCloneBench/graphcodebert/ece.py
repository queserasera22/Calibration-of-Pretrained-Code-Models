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

from model import Model
from main import set_seed, TextDataset
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel

import sys
sys.path.append("../../")
from ece_utils import draw_reliability_graph

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer):
    model_dir = "/".join(args.model_path.split("/")[:-1])

    print("Evaluate-{}...".format(args.test_data_file))
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
            (inputs_ids_1, position_idx_1, attn_mask_1,
             inputs_ids_2, position_idx_2, attn_mask_2,
             test_label) = [x.to(args.device) for x in batch]
            output = model(inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2)

            logits.append(output.cpu())
            labels.append(test_label.cpu())

    end_time = time.time()  # 程序结束时间

    logits = torch.cat(logits, 0)
    labels = torch.cat(labels, 0)
    y_preds = torch.argmax(logits, dim=1)

    print('Test time: %.3fms' % ((end_time - start_time) / len(eval_dataloader) * 1000))

    out_data = {'logits': logits, 'labels': labels, 'preds': y_preds}
    with open(os.path.join(model_dir, "test_pred.pkl"), 'wb') as f:
        pickle.dump(out_data, f)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
    draw_reliability_graph(logits, labels, 10, save_path, args.save_name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--model_path", default="models/codebert_POJ104/model.pt", type=str,
                        help="The path to the fine-tuned model")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")


    parser.add_argument("--test_data_file", default="../dataset/test.jsonl", type=str, required=True,
                        help="The input testing data file (a json file).")
    parser.add_argument("--lang", type=str, default='c',
                        help="the language of the dataset")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--save_name", type=str,
                        help="The name of saved figures.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)

    device = torch.device("cuda")
    args.device = device

    set_seed(args.seed)

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



    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 2
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    model = Model(model, config, tokenizer, args)


    model.load_state_dict(torch.load(args.model_path), strict=False)

    evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
