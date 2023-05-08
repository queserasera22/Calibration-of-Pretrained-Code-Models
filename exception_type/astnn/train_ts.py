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

from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

from model import BatchProgramClassifier
from utils import set_seed, get_batch
from temperature_scaling import ModelWithTemperature

import sys
sys.path.append("../../")
from ece_utils import draw_reliability_graph

set_seed(123456)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train_temperature(model, args):


    valid_data_path = os.path.join(args.data_path, 'dev/blocks.pkl')
    batch_size = args.eval_batch_size

    print("Evaluate-{}...".format(valid_data_path))
    valid_data_t = pd.read_pickle(valid_data_path)
    _size = len(valid_data_t)


    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(model)

    # Tune the model temperature, and save the results
    model.set_temperature(valid_data_t)

    save_path = os.path.join(os.path.dirname(args.model_path), 'ts_test')
    os.makedirs(save_path, exist_ok=True)
    model_filename = os.path.join(save_path, 'model.pt')
    torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model saved to %s' % model_filename)

    test_data_path = os.path.join(args.data_path, 'test/blocks.pkl')
    batch_size = args.eval_batch_size

    print("Evaluate-{}...".format(test_data_path))
    test_data_t = pd.read_pickle(test_data_path)
    _size = len(test_data_t)
    model.evaluate(test_data_t, save_path)

    print("Done")



def main():
    parser = argparse.ArgumentParser(description="Set the params to preprocess the dataset")
    parser.add_argument('--data_path', default='./data/', help='the path of the processed data')
    parser.add_argument('--model_path', default='./models/model.bin', help='the path of the model')
    parser.add_argument('--num_labels', type=int, default=20, help='the number of the labels')
    parser.add_argument("--eval_batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=100, help="hidden dimension")
    parser.add_argument("--encode_dim", type=int, default=128, help="encode dimension")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")

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

    train_temperature(model, args)


if __name__ == "__main__":
    main()

