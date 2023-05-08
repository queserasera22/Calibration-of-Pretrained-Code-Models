import argparse
import csv
import os
import random

import numpy as np
import torch

import sys

sys.path.append("..")

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='CNN text classification')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size for training')
parser.add_argument('--save_path', type=str, default='./CNN_Text',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='../../data/vulnerability_prediction/corpus.pt',
                    help='location of the data corpus')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--embed-dim', type=int, default=128,
                    help='number of embedding dimension [default: 128]')
parser.add_argument('--kernel-num', type=int, default=128,
                    help='number of each kind of kernel')
parser.add_argument('--filter-sizes', type=str, default='3,4,5',
                    help='filter sizes')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_true',
                    help='enables cuda')

args = parser.parse_args()
set_seed(args.seed)

use_cuda = torch.cuda.is_available() and args.cuda_able

# ##############################################################################
# Load data
################################################################################
from dataset import CNNDataset
from torch.utils.data import DataLoader

# from data_loader import DataLoader

data = torch.load(args.data)
args.max_len = data["max_len"]
args.vocab_size = data['dict']['vocab_size']
args.label_size = data['dict']['label_size']
args.filter_sizes = list(map(int, args.filter_sizes.split(",")))

train_data = CNNDataset(
    data['train']['src'],
    data['train']['label'],
    args.max_len,
    cuda=True,
)
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

valid_data = CNNDataset(
    data['valid']['src'],
    data['valid']['label'],
    args.max_len,
    cuda=True,
)
valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

# ##############################################################################
# Build model
# ##############################################################################
from model import CNN_Text

cnn = CNN_Text(args)
if use_cuda:
    cnn = cnn.cuda()
device = torch.device('cuda' if use_cuda else 'cpu')

optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm


def evaluate():
    cnn.eval()
    corrects = eval_loss = 0
    _size = len(valid_data)
    # _size = valid_dataloader.sents_size
    for data, label in tqdm(valid_dataloader, mininterval=0.2,
                            desc='Evaluate Processing', leave=False):
        data = data.to(device)
        label = label.to(device)
        loss, pred = cnn(data, label)

        eval_loss += loss.data
        corrects += (torch.max(pred, 1)
                     [1].view(label.size()).data == label.data).sum()

    return eval_loss.item() / _size, corrects, float(corrects) / _size, _size


def train():
    cnn.train()
    corrects = total_loss = 0
    count = 0
    _size = len(train_data)
    # _size = train_dataloader.sents_size

    predicts = []
    for data, label in tqdm(train_dataloader, mininterval=1,
                            desc='Train Processing', leave=False):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        count = count + 1
        loss, target = cnn(data, label)

        predicts.extend([item.item() for item in target.argmax(dim=1)])

        loss.backward()
        optimizer.step()

        total_loss += loss.data
        corrects += (torch.max(target, 1)
                     [1].view(label.size()).data == label.data).sum()

    return total_loss.item() / _size, float(corrects) / _size, predicts


# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)

    os.makedirs(args.save_path, exist_ok=True)

    header = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']

    with open(args.save_path + "/results.csv", 'w', encoding='UTF8', newline='') as f1:
        writer1 = csv.writer(f1)
        writer1.writerow(header)
        with open(args.save_path + "/predicts.csv", 'w', encoding='UTF8', newline='') as f2:
            writer2 = csv.writer(f2)
            writer2.writerow(data['train']['label'])

            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train_loss, train_acc, outcome = train()

                writer2.writerow(list(outcome))
                print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(
                    epoch, time.time() - epoch_start_time, train_loss))

                valid_loss, corrects, valid_acc, size = evaluate()

                epoch_start_time = time.time()
                print('-' * 80)
                print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(
                    epoch, time.time() - epoch_start_time, valid_loss, valid_acc, corrects, size))
                if not best_acc or best_acc < corrects:
                    best_acc = corrects
                    model_state_dict = cnn.state_dict()
                    model_source = {
                        "settings": args,
                        "model": model_state_dict,
                        "src_dict": data['dict']['train']
                    }
                    torch.save(model_source, args.save_path + '/model.pt')
                    print("Best accuracy is found and the model is saved")

                print('-' * 80)
                result = [epoch, train_loss, train_acc, valid_loss, valid_acc]
                writer1.writerow(result)




except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early | cost time: {:5.2f}min".format(
        (time.time() - total_start_time) / 60.0))
