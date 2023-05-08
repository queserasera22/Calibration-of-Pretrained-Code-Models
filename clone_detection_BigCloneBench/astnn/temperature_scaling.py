import os
import pickle

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from utils import get_batch


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        batch_size = 32

        with torch.no_grad():
            for i in tqdm(range(0, len(valid_loader), batch_size), desc="train"):
                batch = get_batch(valid_loader, i, batch_size)
                input1, inputs2, label = batch
                label = label.cuda()
                self.model.batch_size = len(label)
                self.model.hidden = self.model.init_hidden()
                logits = self.model(input1, inputs2)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.4f, ECE: %.4f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.02, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.4f' % self.temperature.item())
        print('After temperature - NLL: %.4f, ECE: %.4f' % (after_temperature_nll, after_temperature_ece))

        return self

    def evaluate(self, test_loader, model_dir):
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()
        logits_list = []
        labels_list = []

        self.model.eval()

        batch_size = 32
        with torch.no_grad():
            for i in tqdm(range(0, len(test_loader), batch_size), desc="evaluation"):
                batch = get_batch(test_loader, i, batch_size)
                input1, input2, label = batch
                label = label.cuda()

                self.model.batch_size = len(label)
                self.model.hidden = self.model.init_hidden()

                logits = self.model(input1, input2)
                logits_list.append(logits)
                labels_list.append(label)

            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).item() / len(labels)
        print('Accuracy: %.4f' % acc)

        out_data = {'logits': logits, 'labels': labels, 'preds': preds}
        with open(os.path.join(model_dir, "test_pred.pkl"), 'wb') as f:
            pickle.dump(out_data, f)

        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('After temperature - NLL: %.4f, ECE: %.4f' % (after_temperature_nll, after_temperature_ece))


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        acc = torch.sum(predictions == labels).item() / len(labels)
        print('Accuracy: %.4f' % acc)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece