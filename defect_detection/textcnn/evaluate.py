import os
import pickle
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile
import argparse
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import sys
sys.path.append("..")


from model import CNN_Text
from dataset import CNNDataset

parser = argparse.ArgumentParser(description='CNN classification evaluation')
parser.add_argument('--model_path', type=str, default='./models/model.pt',
                    help='the path of trained cnn model[default: ./models/model.pt]')
parser.add_argument('--data_path', type=str, default='./data/corpus.pt',
                    help='the path of preprocessed test data[default: ./data/corpus.pt]')

params = parser.parse_args()

# model_source = {
#     "settings": args,
#     "model": model_state_dict,
#     "src_dict": data['dict']['train']
# }


def evaluate(save_path, data_path):
    data = torch.load(data_path)
    model_source = torch.load(save_path)
    args = model_source["settings"]

    cnn = CNN_Text(args)

    total_params = sum(p.numel() for p in cnn.parameters())
    print(f"{total_params:,} total parameters.")
    print(f"{total_params*4/1e6} MB model size")
    cnn.load_state_dict(model_source["model"])

    total_params = sum(p.numel() for p in cnn.parameters())
    print(f"{total_params:,} total parameters.")
    print(f"{total_params*4/1e6} MB model size")

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
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    _size = len(test_data)

    total_acc_test = 0

    start_time = time.time()  # 程序开始时间
    cnn.eval()


    labels_all = []
    logits_all = []
    for data, label in tqdm(test_dataloader, mininterval=0.2,
                            desc='Evaluate Processing', leave=False):
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = cnn(data)

            logits_all.append(output.cpu())
            labels_all.append(label.cpu())

    labels_all = torch.cat(labels_all, dim=0)
    logits_all = torch.cat(logits_all, dim=0)
    predict_all = torch.argmax(logits_all, dim=1)

    accuracy = accuracy_score(labels_all, predict_all)
    recall = recall_score(labels_all, predict_all)
    precision = precision_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all)
    results = {
        "eval_acc": float(accuracy),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1)
    }
    print(results)

    end_time = time.time()  # 程序结束时间
    print('Test time: %.3fms' % ((end_time - start_time) / _size * 1000))

    model_dir = "/".join(save_path.split("/")[:-1])
    out_data = {'logits': logits_all, 'labels': labels_all, 'preds': predict_all}
    with open(os.path.join(model_dir, "test_pred.pkl"), 'wb') as f:
        pickle.dump(out_data, f)

    # 效率评估 time + FLOPs
    input = torch.randint(1, 1024, (1, 512)).cuda()
    flops, params = profile(cnn, inputs=(input,))
    # flops, params = get_model_complexity_info(model, (1, 256), input_constructor=input_constructor, as_strings=True,
    #                                          print_per_layer_stat=False, verbose=True)
    print('flops is %.3fM' % (flops / 1e6))  # 打印计算量
    print('params is %.3fM' % (params / 1e6))  # 打印参数


evaluate(params.model_path, params.data_path)
