import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class Model(nn.Module):
    def __init__(self, encoder, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.args = args
        
    def forward(self, input_ids=None, labels=None): 
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        # prob = F.softmax(torch.tensor(logits), dim=-1)
        prob = logits
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob