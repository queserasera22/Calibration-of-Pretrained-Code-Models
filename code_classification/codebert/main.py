import os
import pickle
import time

import pandas as pd
import torch
import logging
import argparse
import warnings
import numpy as np

from tqdm import tqdm
from model import Model
from utils import set_seed, TextDataset
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaForSequenceClassification, \
    RobertaTokenizer
import torch.nn.functional as F

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train(args, model, tokenizer):
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters(
        ) if not any(nd in n for nd in no_decay)]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0
    model.zero_grad()

    for idx in range(0, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, _ = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, eval_when_training=True)

                    logger.info("  " + "*" * 20)
                    logger.info("  Current ACC:%s", round(results["eval_acc"], 4))
                    logger.info("  Best ACC:%s", round(best_acc, 4))
                    logger.info("  " + "*" * 20)

                    if results["eval_acc"] >= best_acc:
                        best_acc = results["eval_acc"]


                        output_dir = os.path.join(args.output_dir, args.dataset)

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        output_dir = os.path.join(output_dir, "{}".format("model.bin"))
                        torch.save(model.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        logger.info("Model checkpoint are not saved")


def evaluate(args, model, tokenizer, eval_when_training=False, is_write_preds=False):
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits = []
    labels = []
    preds = []
    time_count = []
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        bar.set_description("evaluation")
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            time_start = time.time()
            logit = model(inputs)
            time_end = time.time()
            logit = logit.cpu().numpy()
            time_count.append(time_end - time_start)
            logits.append(logit)
            preds.append(logit.argmax(axis=1))
        labels.append(label.cpu().numpy())
    print(sum(time_count) / len(time_count))
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = np.concatenate(preds, 0)

    # logits = torch.nn.functional.softmax(logits)
    eval_acc = np.mean(labels == preds)
    recall = recall_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    result = {
        "eval_acc": eval_acc,
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


    if is_write_preds:
        out_data = {'logits': logits, 'labels': labels, 'preds': preds}
        with open(os.path.join(args.output_dir, args.dataset, "test_pred.pkl"), 'wb') as f:
            pickle.dump(out_data, f)
    # eval_pred_df = pd.DataFrame({'label': labels, 'pred': preds})
    # eval_pred_df.to_csv(os.path.join(args.output_dir, "checkpoint", args.dataset, '{}'.format('eval_pred.csv')))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/codebert-base", type=str, required=True,
                        help="The name or the path of the used model. [default: microsoft/codebert-base ]")
    parser.add_argument("--dataset", type=str, default='POJ104',
                        help="the dataset model trained, choose among ['POJ104', 'Java250', 'Python800']")
    parser.add_argument("--num_labels", type=int, required=True,
                        help="the number of classes.")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="../", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--epoch", type=int, default=42,
                        help="epoch for training")
    parser.add_argument("--patience", type=int, default=5,
                        help="patience for early stopping")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="label smoothing")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)
    # args.device = torch.device("cpu")
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu

    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    set_seed(args.seed)

    config = RobertaConfig.from_pretrained(args.model_name)
    config.num_labels = args.num_labels

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.do_lower_case = True

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = Model(RobertaForSequenceClassification.from_pretrained(args.model_name, config=config), args)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train(args, model, tokenizer)

    if args.do_eval:
        params = sum(p.numel() for p in model.parameters())
        logger.info("size %f", params)
        logger.info(f"{params * 4 / 1e6} MB model size")
        output_dir = os.path.join(args.output_dir, args.dataset, 'model.bin')
        model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(
            output_dir).items()}, strict=False)
        params = sum(p.numel() for p in model.parameters())
        logger.info("size %f", params)
        logger.info(f"{params * 4 / 1e6} MB model size")
        model.to(args.device)
        evaluate(args, model, tokenizer, is_write_preds=True)


if __name__ == "__main__":
    main()
