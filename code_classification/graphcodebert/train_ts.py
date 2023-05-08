import argparse
import pickle
import time
import os
import warnings
import logging

import torch

from model import Model
from main import set_seed, TextDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel

import sys
sys.path.append("../../")
from temperature_scaling import ModelWithTemperature

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer):
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

    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)
    model.evaluate(eval_dataloader, save_path)


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

    parser.add_argument("--num_labels", default=104, type=int, help="The number of labels")

    parser.add_argument("--eval_data_file", default="../dataset/valid.jsonl", type=str, required=True,
                        help="The input validating data file (a json file).")
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

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = args.num_labels
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    model = Model(model, config, tokenizer, args)


    model.load_state_dict(torch.load(args.model_path), strict=False)

    evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
