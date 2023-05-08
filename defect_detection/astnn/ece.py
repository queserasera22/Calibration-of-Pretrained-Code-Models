import os
import pickle

import numpy as np
import pandas as pd
import torch
import argparse
import time

import warnings
import random
import logging
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("")

from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

from model import BatchProgramClassifier
from utils import set_seed, get_batch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import sys
sys.path.append("../../")
from ece_utils import draw_reliability_graph

random.seed(123456)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)



def evaluate(model, args):

    model_dir = "/".join(args.model_path.split("/")[:-1])
    if os.path.exists(os.path.join(model_dir, "test_pred.pkl")):
        print("Load test_pred.pkl")
        with open(os.path.join(model_dir, "test_pred.pkl"), 'rb') as f:
            loaded_tensors = pickle.load(f)
        labels = loaded_tensors['labels']
        predicts = loaded_tensors['logits']
    else:
        test_data_path = os.path.join(args.data_path, 'test/blocks.pkl')
        batch_size = args.eval_batch_size

        print("Evaluate-{}...".format(test_data_path))
        test_data_t = pd.read_pickle(test_data_path)
        _size = len(test_data_t)

        model.eval()

        predicts = []
        labels = []
        y_preds = []
        start_time = time.time()  # 程序开始时间
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data_t), batch_size), desc="evaluation"):
                batch = get_batch(test_data_t, i, batch_size)
                test_inputs, test_labels = batch
                test_labels = test_labels.to(args.device)

                model.batch_size = len(test_labels)
                model.hidden = model.init_hidden()
                eval_loss, output = model(test_inputs, test_labels)

                predicts.append(output.cpu())
                labels.append(test_labels.cpu())
                y_preds.append(torch.argmax(output, dim=1).cpu())

        end_time = time.time()  # 程序结束时间
        predicts = torch.cat(predicts, 0)
        labels = torch.cat(labels, 0)
        y_preds = torch.cat(y_preds, 0)

        print("Test accuracy: %.4f" % (torch.sum(y_preds == labels).item() / _size))
        print('Test time: %.3fms' % ((end_time - start_time) / _size * 1000))

        out_data = {'logits': predicts, 'labels': labels, 'preds': y_preds}
        with open(os.path.join(model_dir, "test_pred.pkl"), 'wb') as f:
            pickle.dump(out_data, f)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
    draw_reliability_graph(predicts, labels, 10, save_path, args.save_name)



def main():
    parser = argparse.ArgumentParser(description="Set the params to preprocess the dataset")
    parser.add_argument('--data_path', default='../dataset/POJ104', help='the path of the processed data')
    parser.add_argument('--model_path', default='./model/POJ104', help='the path of the model')
    parser.add_argument('--num_labels', type=int, default=104, help='the number of the labels')
    parser.add_argument("--eval_batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=100, help="hidden dimension")
    parser.add_argument("--encode_dim", type=int, default=128, help="encode dimension")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--save_name", type=str,
                        help="The name of saved figures.")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    word2vec = Word2Vec.load(os.path.join(args.data_path, "train/embedding/node_w2v_128")).wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = args.hidden_dim
    ENCODE_DIM = args.encode_dim
    LABELS = args.num_labels
    BATCH_SIZE = args.eval_batch_size

    USE_GPU = torch.cuda.is_available() and not args.no_cuda

    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                                   USE_GPU, embeddings)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)

    evaluate(model, args)


if __name__ == "__main__":
    main()

