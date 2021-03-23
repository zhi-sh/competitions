# -*- coding: utf-8 -*-
# @DateTime :2021/3/16
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
from torch import nn
from transformers import BertForPreTraining


class BertModel(nn.Module):
    def __init__(self, model: BertForPreTraining, output_size):
        super(BertModel, self).__init__()
        self.model = model.bert
        self.transform = model.cls.predictions.transform
        self.linear = nn.Linear(in_features=768, out_features=output_size)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = x.pooler_output
        out = self.linear(pooled_output)
        return out
