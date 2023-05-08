import os
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F

def calc_ece(logits, labels, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    logits = torch.tensor(logits).clone().detach()
    labels = torch.tensor(labels).clone().detach()
    print(labels.device)

    # logits = F.softmax(logits, dim=-1)

    # # 对sigmoid函数（1分类）
    # tem = logits[:, 0]
    # softmax_max = (tem > 0.5) * tem + (tem <= 0.5) * (1 - tem)
    # predictions = tem > 0.5

    # 对softmax函数（多分类）
    softmax_max, predictions = torch.max(logits, 1)
    correctness = predictions.eq(labels)

    print("accuracy: ", (predictions == labels).sum().item() / len(labels))


    ece = torch.zeros(1, device=logits.device)
    bin_accs = torch.zeros(bins)
    bin_confs = torch.zeros(bins)
    bin_sizes = torch.zeros(bins)

    acc = torch.zeros(1, device=logits.device)

    for i, bin_lower, bin_upper in zip(range(bins), bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        bin_sizes[i] = prop_in_bin

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()
            bin_accs[i] = accuracy_in_bin
            bin_confs[i] = avg_confidence_in_bin

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            acc += accuracy_in_bin * prop_in_bin

    print("Acc {} ".format(acc.item()))
    print("Acc {0:.2f} ".format(acc.item() * 100))

    print("ECE {} ".format(ece.item()))
    print("ECE {0:.2f} ".format(ece.item() * 100))

    return ece.item(), acc.item(), bin_boundaries, bin_accs, bin_confs, bin_sizes


def draw_reliability_graph(preds, labels, num_bins, save_path, save_name):
    ECE, ACC, bins, bin_accs, bin_confs, bin_sizes = calc_ece(preds, labels, num_bins)

    # exit(0)

    bins = bins[1:]

    print(bins)
    print(bin_confs)
    print(bin_accs)
    print(bin_sizes)

    index_x = [b - 0.05 for b in bins]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # plt.xlabel('Confidence')
    # plt.ylabel('Frac.')

    # Error bars
    plt.bar(index_x, bin_sizes, width=0.1, alpha=1, edgecolor='black', color='b')
    for i, j in zip(index_x, bin_sizes):
        plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=16)


    plt.tick_params(labelsize=20)

    os.makedirs(save_path, exist_ok=True )

    plt.savefig(os.path.join(save_path, '{}-frac.png'.format(save_name)), bbox_inches='tight')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # plt.xlabel('Confidence')
    # plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    # # Error bars
    # plt.bar(index_x, index_x, width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\', label='Gap')

    # Draw bars and identity line
    plt.bar(index_x, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b', label='Output')

    gaps = [a - b for a, b in zip(bin_confs, bin_accs)]
    plt.bar(index_x, gaps, bottom=bin_accs, width=0.1, alpha=0.3, edgecolor='r', color='r', hatch='\\', label='Gap')

    # plt.bar(index_x, gaps, bottom=bin_accs, color='b', alpha=0.5, width=0.1, hatch='//', edgecolor='r')
    # plt.bar(index_x, gaps, bottom=bin_accs, color=[1, 0.7, 0.7], alpha=0.5, width=0.1,  hatch='//', edgecolor='r')

    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # Gap and output legend
    plt.legend(fontsize=21)

    textstr = 'ECE = {:.2f}%'.format(ECE * 100) + '\n' + 'ACC = {:.2f}%'.format(ACC * 100)
    props = dict(boxstyle='round', alpha=0.5, facecolor='white', edgecolor='black')
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=25,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tick_params(labelsize=20)

    # plt.show()
    plt.savefig(os.path.join(save_path, '{}-acc.png'.format(save_name)), bbox_inches='tight')