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

    forget_record_df = pd.DataFrame()
    performance_record_df = pd.DataFrame()

    for idx in range(0, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0

        # train_logits = []
        # train_labels = []
        # train_indice = []

        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            indexes = batch[2].to(args.device)
            model.train()
            loss, logit = model(inputs, labels)

            # # 记录每一步的训练结果
            # train_logits.append(logit.detach().cpu().numpy())
            # train_labels.append(labels.detach().cpu().numpy())
            # train_indice.append(indexes.detach().cpu().numpy())

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
                # avg_loss = round(
                #     np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                #
                # if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     logging_loss = tr_loss
                #     tr_nb = global_step

                # 一个epoch 记录一次训练集的结果，以统计遗忘情况 ， 其中args.save_steps=len(train_dataloader)
                # if args.save_steps > 0 and global_step % args.save_steps == 0:
                #
                #     preds, labels, indice = evaluate(args, model, tokenizer, args.train_data_file,
                #                              eval_when_training=True, return_preds=True)
                #     epochs = [idx] * len(indice)
                #     epoch_train_df = pd.DataFrame({'idx': indice, 'label': labels, 'epoch': epochs, 'pred': preds})
                #     epoch_train_df['pred'] = epoch_train_df["pred"].astype(int)
                #     forget_record_df = pd.concat([forget_record_df, epoch_train_df])
                #     checkpoint_prefix = 'checkpoint-best-acc'
                #     output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     forget_record_df.to_csv(os.path.join(output_dir, '{}'.format('train_forget.csv')))

                # 一个epoch 记录5次训练集和验证集上的performance，并保存模型， 其中args.save_steps=len(train_dataloader)
                if args.save_steps > 0 and global_step % int(args.save_steps) == 0:
                    if args.evaluate_during_training:

                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        train_results = evaluate(args, model, tokenizer, args.train_data_file,
                                                                        eval_when_training=True)
                        results = evaluate(args, model, tokenizer, args.eval_data_file, eval_when_training=True)
                        # logits = np.concatenate(train_logits, 0)
                        # labels = np.concatenate(train_labels, 0)
                        # preds = logits[:, 0] > 0.5
                        performance_record_df = performance_record_df.append({
                            'train_loss': train_results['eval_loss'],
                            'train_accuracy': train_results['eval_acc'],
                            'train_f1': train_results['eval_f1'],
                            'train_precision': train_results['eval_precision'],
                            'train_recall': train_results['eval_recall'],
                            'valid_loss': results['eval_loss'],
                            'valid_accuracy': results['eval_acc'],
                            'valid_f1': results['eval_f1'],
                            'valid_precision': results['eval_precision'],
                            'valid_recall': results['eval_recall'],
                            'epoch': idx,
                            'step': global_step
                        }, ignore_index=True)
                        performance_record_df.to_csv(os.path.join(output_dir, '{}'.format('model_performance.csv')))

                        logger.info("  " + "*" * 20)
                        logger.info("  Current ACC:%s", round(results["eval_acc"], 4))
                        logger.info("  Best ACC:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)

                        if results["eval_acc"] >= best_acc:
                            best_acc = results["eval_acc"]
                            output_dir = os.path.join(output_dir, "{}".format("model.bin"))
                            torch.save(model.state_dict(), output_dir)
                            logger.info("Saving model checkpoint to %s", output_dir)
                        else:
                            logger.info("Model checkpoint are not saved")


def evaluate(args,  model, tokenizer, eval_data_file, eval_when_training=False, return_preds=False, save_logits=False):
    eval_dataset = TextDataset(tokenizer, args, eval_data_file)
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
    indexes = []
    time_count = []
    eval_loss = 0.0
    nb_eval_steps = 0

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        bar.set_description("evaluation")
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        index = batch[2].to(args.device)
        with torch.no_grad():
            time_start = time.time()
            lm_loss, logit = model(inputs, label)
            time_end = time.time()

            eval_loss += lm_loss.mean().item()
            time_count.append(time_end - time_start)

        nb_eval_steps += 1

        logits.append(logit.cpu().numpy())
        labels.append(label.cpu().numpy())
        indexes.append(index.cpu().numpy())

    print(sum(time_count) / len(time_count))
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    indexes = np.concatenate(indexes, 0)

    preds = np.argmax(logits, axis=1)
    # preds = logits[:, 0] > 0.5
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_acc = np.mean(labels == preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": eval_acc,
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    if save_logits:
        checkpoint_prefix = 'checkpoint-best-acc/test_pred.pkl'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        out_data = {'logits': logits, 'labels': labels, 'preds': preds}
        with open(output_dir, 'wb') as f:
            pickle.dump(out_data, f)

    if return_preds:
        return preds, labels, indexes

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/codebert-base", type=str, required=True,
                        help="The name or the path of the used model. [default: microsoft/codebert-base ]")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./models/", type=str,
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
                        help="random seed for initialization")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu

    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    set_seed(args.seed)

    config = RobertaConfig.from_pretrained(args.model_name)
    config.num_labels = 2

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.do_lower_case = True

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = Model(RobertaForSequenceClassification.from_pretrained(args.model_name, config=config))

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train(args, model, tokenizer)

    if args.do_eval:
        params = sum(p.numel() for p in model.parameters())
        logger.info("size %f", params)


        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))

        model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(
            output_dir).items()}, strict=False)
        params = sum(p.numel() for p in model.parameters())
        logger.info("size %f", params)
        model.to(args.device)
        evaluate(args, model, tokenizer, args.eval_data_file, save_logits=True)


if __name__ == "__main__":
    main()
