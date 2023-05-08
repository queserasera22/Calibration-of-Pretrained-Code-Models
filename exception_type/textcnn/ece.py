import os
import pickle

import torch
import argparse
import time

import warnings
import random
import logging
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import CNN_Text
from dataset import CNNDataset
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import sys
sys.path.append("../../")
from ece_utils import draw_reliability_graph

random.seed(123456)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def evaluate(args):
    model_dir = "/".join(args.model_path.split("/")[:-1])

    data = torch.load(args.data_path)
    model_source = torch.load(args.model_path)
    settings = model_source["settings"]

    model = CNN_Text(settings)
    model.load_state_dict(model_source["model"])

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)

    test_data = CNNDataset(
        data['test']['src'],
        data['test']['label'],
        settings.max_len,
        cuda=True,
    )
    test_dataloader = DataLoader(test_data, batch_size=settings.batch_size, shuffle=False)
    _size = len(test_data)

    model.eval()

    logits = []
    labels = []
    start_time = time.time()  # 程序开始时间
    with torch.no_grad():
        for test_input, test_label in tqdm(test_dataloader):
            test_label = test_label.to(device)
            input_id = test_input.squeeze(1).to(device)
            output = model(input_id)

            logits.append(output.cpu())
            labels.append(test_label.cpu())

    end_time = time.time()  # 程序结束时间

    logits = torch.cat(logits, 0)
    labels = torch.cat(labels, 0)
    preds = torch.argmax(logits, dim=1)

    out_data = {'logits': logits, 'labels': labels, 'preds': preds}
    with open(os.path.join(model_dir, "test_pred.pkl"), 'wb') as f:
        pickle.dump(out_data, f)
    # if os.path.exists(os.path.join(model_dir, "test_pred.pkl")):
    #     # print("Load test_pred.pkl")
    #     # with open(os.path.join(model_dir, "test_pred.pkl"), 'rb') as f:
    #     #     loaded_tensors = pickle.load(f)
    #     # labels = loaded_tensors['labels']
    #     # logits = loaded_tensors['logits']
    # else:


    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
    draw_reliability_graph(logits, labels, 10, save_path, args.save_name)


def main():
    parser = argparse.ArgumentParser(description='CNN code classification evaluation')
    parser.add_argument('--model_path', type=str, default='./outputs/CNN_POJ104.pt',
                        help='the path of trained cnn model[default: ./outputs/CNN_POJ104.pt]')
    parser.add_argument('--data_path', type=str, default='./dataset/POJ104.pt',
                        help='the path of preprocessed test data[default: ./dataset/POJ104.pt]')
    parser.add_argument("--save_name", type=str,
                        help="The name of saved figures.")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)

    evaluate(args)


if __name__ == "__main__":
    main()

