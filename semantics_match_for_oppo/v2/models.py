# -*- coding: utf-8 -*-
# @DateTime :2021/3/19
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch
from torch import nn
from transformers import BertConfig, BertModel


class PretrainedBertModel(nn.Module):
    def __init__(self, embeddings_size, checkpoint_path, embeddings_dim=768, max_len=64, keep_tokens=None):
        super(PretrainedBertModel, self).__init__()
        # self.bert = BertModel(BertConfig(embeddings_size))
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lr = nn.Linear(in_features=embeddings_dim, out_features=768)
        self.layer_norm = nn.LayerNorm((max_len, embeddings_dim))
        self.out = nn.Linear(in_features=768, out_features=embeddings_size)

        # 使用bert的预训练embedding作为初始化embedding
        if keep_tokens is not None:
            self.embedding = nn.Embedding(embeddings_size, embeddings_dim)
            weight = torch.load(checkpoint_path)
            weight = nn.Parameter(weight['weight'][keep_tokens])
            self.embedding.weight = weight
            self.bert.embeddings.word_embeddings = self.embedding

    def forward(self, x):
        x = self.bert(**x)
        x = self.lr(x['last_hidden_state'])
        x = self.layer_norm(x)
        out = self.out(x)
        return out
