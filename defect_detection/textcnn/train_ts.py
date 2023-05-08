import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

import sys
sys.path.append("../../")
from temperature_scaling import ModelWithTemperature
from model import CNN_Text
from dataset import CNNDataset

parser = argparse.ArgumentParser(description='CNN code classification evaluation')
parser.add_argument('--model_path', type=str, default='./models/test/model.pt',
                    help='the path of trained cnn model[default: ./models/model.pt]')
parser.add_argument('--data_path', type=str, default='./data/corpus.pt',
                    help='the path of preprocessed test data[default: ./data/corpus.pt]')

parser_args = parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(123456)


def train_temperature(model_path, data_path):
    data = torch.load(data_path)
    model_source = torch.load(model_path)
    args = model_source["settings"]
    print(args)

    cnn = CNN_Text(args)
    cnn.load_state_dict(model_source["model"])

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        cnn = cnn.cuda()

    valid_data = CNNDataset(
        data['valid']['src'],
        data['valid']['label'],
        args.max_len,
        cuda=True,
    )
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    print('valid_data size: %d' % len(valid_data))

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(cnn)

    # Tune the model temperature, and save the results
    model.set_temperature(valid_dataloader)

    save_path = os.path.join(os.path.dirname(model_path), 'ts_test')
    os.makedirs(save_path, exist_ok=True)
    model_filename = os.path.join(save_path, 'model.pt')
    torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model saved to %s' % model_filename)

    test_data = CNNDataset(
        data['test']['src'],
        data['test']['label'],
        args.max_len,
        cuda=True,
    )
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    print('test_data size: %d' % len(test_data))

    model.evaluate(test_dataloader)

    print('Done!')

def evaluate_temperature(model_path, data_path):
    data = torch.load(data_path)
    model_source = torch.load(model_path)
    args = model_source["settings"]
    print(args)

    cnn = CNN_Text(args)
    cnn.load_state_dict(model_source["model"])

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        cnn = cnn.cuda()

    test_data = CNNDataset(
        data['test']['src'],
        data['test']['label'],
        args.max_len,
        cuda=True,
    )
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    print('test_data size: %d' % len(test_data))

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(cnn)
    save_path = os.path.join(os.path.dirname(model_path), 'ts', 'model.pt')
    model.load_state_dict(torch.load(save_path))
    model.evaluate(test_dataloader)
    print('Done!')

if __name__ == '__main__':
    train_temperature(parser_args.model_path, parser_args.data_path)
    # evaluate_temperature(parser_args.model_path, parser_args.data_path)

