import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


class RobertaClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features):
        x = features[:, 0, :]
        x = x.reshape(-1, x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, input_ids=None, labels=None):

        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        # prob = F.softmax(torch.tensor(logits), dim=-1)
        prob = torch.tensor(logits)

        if labels is not None:
            labels = labels.squeeze(-1)
            loss_fct = CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
