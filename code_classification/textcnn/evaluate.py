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

params = parser.parse_args()

# save_path = './outputs/CNN_POJ104.pt'
# data_path = './dataset/POJ104.pt'
# save_path = './outputs/CNN_Java250.pt'
# data_path = './dataset/Java250.pt'
# save_path = './models/CNN_Python800.pt'
# data_path = './data/Python800.pt'

# model_source = {
#     "settings": args,
#     "model": model_state_dict,
#     "src_dict": data['dict']['train']
# }


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

    total_acc_test = 0

    start_time = time.time()  # 程序开始时间
    cnn.eval()
    for data, label in tqdm(test_dataloader, mininterval=0.2,
                            desc='Evaluate Processing', leave=False):
        data = data.to(device)
        label = label.to(device)
        pred = cnn(data)
        acc = (pred.argmax(dim=1) == label).sum().item()
        total_acc_test += acc

    end_time = time.time()  # 程序结束时间

    print(f'Test Accuracy: {float(total_acc_test) / _size * 100 : .3f}%')
    print('Test time: %.3fms' % ((end_time - start_time) / _size * 1000))

    # 效率评估 time + FLOPs
    input = torch.randint(1, 1024, (1, 512)).cuda()
    flops, params = profile(cnn, inputs=(input,))
    # flops, params = get_model_complexity_info(model, (1, 256), input_constructor=input_constructor, as_strings=True,
    #                                          print_per_layer_stat=False, verbose=True)
    print('flops is %.3fM' % (flops / 1e6))  # 打印计算量
    print('params is %.3fM' % (params / 1e6))  # 打印参数


evaluate(params.model_path, params.data_path)
