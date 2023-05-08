import pickle

import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

from model import BatchProgramClassifier
from utils import set_seed, get_batch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
import argparse
import logging
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)


def evaluate(data_path, batch_size):
    print("Evaluate-{}...".format(data_path))
    test_data_t = pd.read_pickle(data_path)

    # testing procedure
    logits = []
    trues = []
    eval_total_loss = 0.0
    eval_total = 0.0
    model.eval()

    for i in tqdm(range(0, len(test_data_t), batch_size), desc="evaluation"):
        batch = get_batch(test_data_t, i, batch_size)
        test_inputs, test_labels = batch
        if USE_GPU:
            test_labels = test_labels.cuda()

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()

        with torch.no_grad():
            eval_loss, output = model(test_inputs, Variable(test_labels))
            # calc testing acc
            logits.append(output.data.cpu())
            trues.append(test_labels.cpu())

        eval_total += len(test_labels)
        eval_total_loss += eval_loss.item() * len(test_labels)

    avg_eval_loss = eval_total_loss / eval_total

    logits = torch.cat(logits, 0)
    predicts = torch.argmax(logits, 1)
    trues = torch.cat(trues, 0)

    logger.info("Evaluate loss: {}".format(avg_eval_loss))

    eval_accuracy = accuracy_score(trues, predicts)
    eval_recall = recall_score(trues, predicts, average='macro')
    eval_precision = precision_score(trues, predicts, average='macro')
    eval_f1 = f1_score(trues, predicts, average='macro')
    result = {
        "eval_acc": float(eval_accuracy),
        "eval_recall": float(eval_recall),
        "eval_precision": float(eval_precision),
        "eval_f1": float(eval_f1)
    }

    model_dir = "/".join(args.model_path.split("/")[:-1])
    out_data = {'logits': logits, 'labels': trues, 'preds': predicts}
    with open(os.path.join(model_dir, "test_pred.pkl"), 'wb') as f:
        pickle.dump(out_data, f)

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    return avg_eval_loss, result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Set the params to preprocess the dataset")
    parser.add_argument('--data_path', default='../dataset/POJ104', help='the path of the processed data')
    parser.add_argument('--model_path', default='./model/POJ104', help='the path of the model')
    parser.add_argument('--num_labels', type=int, default=104, help='the number of the labels')
    parser.add_argument("--train_batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=100, help="hidden dimension")
    parser.add_argument("--encode_dim", type=int, default=128, help="encode dimension")
    parser.add_argument("--epochs", type=int, default=15, help="epochs")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to run test.")
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    save_path = args.model_path
    os.makedirs(save_path, exist_ok=True)

    train_data = pd.read_pickle(os.path.join(args.data_path, 'train/blocks.pkl'))
    word2vec = Word2Vec.load(os.path.join(args.data_path, "train/embedding/node_w2v_128")).wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = args.hidden_dim
    ENCODE_DIM = args.encode_dim
    LABELS = args.num_labels
    EPOCHS = args.epochs
    BATCH_SIZE = args.train_batch_size
    USE_GPU = torch.cuda.is_available() and not args.no_cuda

    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                                   USE_GPU, embeddings)
    model = model.to(args.device)

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=args.lr)


    if args.do_train:

        best_acc = 0
        not_f1_inc_cnt = 0
        performance_record_df = pd.DataFrame()

        print('Start training...')
        # training procedure
        best_model = model
        for epoch in range(EPOCHS):
            model.train()
            start_time = time.time()

            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            avg_loss = 0.0
            avg_acc = 0.0
            i = 0
            for i in tqdm(range(0, len(train_data), BATCH_SIZE),
                          desc="Training: Epoch {}".format(epoch)):
                batch = get_batch(train_data, i, BATCH_SIZE)
                train_inputs, train_labels = batch

                train_labels = train_labels.to(args.device)

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                loss, output = model(train_inputs, train_labels)

                loss.backward()
                optimizer.step()

                loss = loss.item()

                # calc training acc
                predicted = torch.argmax(output.data, 1)
                # _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == train_labels).sum().item()
                total += len(train_labels)
                total_loss += loss * len(train_inputs)

                # del train_inputs, train_labels, output, loss
                # torch.cuda.empty_cache()


            avg_loss = total_loss / total
            avg_acc = total_acc / total

            eval_loss, eval_results = evaluate(os.path.join(args.data_path, 'dev/blocks.pkl'), args.eval_batch_size)
            end_time = time.time()

            performance_record_df = performance_record_df.append({
                'train_loss': avg_loss,
                'train_accuracy': avg_acc,
                'valid_loss': eval_loss,
                'valid_accuracy': eval_results['eval_acc'],
                'epoch': epoch
            }, ignore_index=True)
            performance_record_df.to_csv(os.path.join(save_path, '{}'.format('model_performance.csv')))

            if eval_results["eval_acc"] >= best_acc:
                not_f1_inc_cnt = 0
                best_acc = eval_results["eval_acc"]
                output_dir = os.path.join(save_path, "{}".format("model.bin"))
                torch.save(model.state_dict(), output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)
            else:
                not_f1_inc_cnt += 1
                logger.info("Model checkpoint are not saved")
                logger.info("ACC does not increase for %d epochs", not_f1_inc_cnt)
                if not_f1_inc_cnt > args.patience:
                    logger.info("Early stop as acc do not increase for %d times", not_f1_inc_cnt)
                    break

    if args.do_test:
        model.load_state_dict(torch.load(os.path.join(save_path, "{}".format("model.bin"))))
        _, test_result = evaluate(os.path.join(args.data_path, 'test/blocks.pkl'), args.eval_batch_size)
        print(test_result)





