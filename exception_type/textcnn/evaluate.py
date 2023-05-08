import os
import pickle
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile
import argparse

import sys
sys.path.append("../../code_classification")


from model import CNN_Text
from dataset import CNNDataset

parser = argparse.ArgumentParser(description='CNN code classification evaluation')
parser.add_argument('--model_path', type=str, default='./outputs/CNN_POJ104.pt',
                    help='the path of trained cnn model[default: ./outputs/CNN_POJ104.pt]')
parser.add_argument('--data_path', type=str, default='./dataset/POJ104.pt',
                    help='the path of preprocessed test data[default: ./dataset/POJ104.pt]')

parser_args = parser.parse_args()

def evaluate(save_path, data_path):
    data = torch.load(data_path)
    model_source = torch.load(save_path)
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
    _size = len(test_data)

    start_time = time.time()  # 程序开始时间
    cnn.eval()
    labels = []
    logits = []

    with torch.no_grad():
        for data, label in tqdm(test_dataloader, mininterval=0.2,
                                desc='Evaluate Processing', leave=False):
            data = data.to(device)
            label = label.to(device)
            pred = cnn(data)
            logits.append(pred.cpu())
            labels.append(label.cpu())

    labels = torch.cat(labels, dim=0)
    logits = torch.cat(logits, dim=0)
    preds = logits.argmax(dim=1)

    total_acc_test = (preds == labels).sum().item()


    end_time = time.time()  # 程序结束时间

    print(f'Test Accuracy: {float(total_acc_test) / _size * 100 : .4f}%')
    print('Test time: %.3fms' % ((end_time - start_time) / _size * 1000))

    # 效率评估 time + FLOPs
    input = torch.randint(1, 1024, (1, 512)).cuda()
    flops, params = profile(cnn, inputs=(input,))
    # flops, params = get_model_complexity_info(model, (1, 256), input_constructor=input_constructor, as_strings=True,
    #                                          print_per_layer_stat=False, verbose=True)
    print('flops is %.3fM' % (flops / 1e6))  # 打印计算量
    print('params is %.3fM' % (params / 1e6))  # 打印参数

    # # 保存预测结果
    # model_dir = "/".join(parser_args.model_path.split("/")[:-1])
    # out_data = {'logits': logits, 'labels': labels, 'preds': preds}
    # with open(os.path.join(model_dir, "test_pred.pkl"), 'wb') as f:
    #     pickle.dump(out_data, f)


evaluate(parser_args.model_path, parser_args.data_path)
