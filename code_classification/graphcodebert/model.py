import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.query = 0

    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels=None):
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)

        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(
            inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :,
                                         None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / \
            (nodes_to_token_mask.sum(-1)+1e-10)[:, :, None]
        avg_embeddings = torch.einsum(
            "abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * \
            (~nodes_mask)[:, :, None]+avg_embeddings*nodes_mask[:, :, None]
        outputs = self.encoder(inputs_embeds=inputs_embeddings,
                               attention_mask=attn_mask, position_ids=position_idx)[0]

        logits = outputs
        prob = torch.softmax(logits, dim=-1)
        # prob = logits
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.05)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
