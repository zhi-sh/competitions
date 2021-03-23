# -*- coding: utf-8 -*-
# @DateTime :2021/3/19
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import json
import random
from collections import defaultdict
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertModel
import settings as conf


def save_bert_embeddings(model_path, model_name="bert-base-chinese"):
    r'''保存预训练的Bert 词向量'''
    if not os.path.exists(model_path):
        model = BertModel.from_pretrained(model_name)
        torch.save(model.embeddings.word_embeddings.state_dict(), model_path)


def build_vocab(train: pd.DataFrame, testA: pd.DataFrame, min_count=5):
    def get_dict(data):
        word_dict = defaultdict(int)
        for i in tqdm(range(data.shape[0])):
            text = data.text_a.iloc[i].split() + data.text_b.iloc[i].split()
            for c in text:
                word_dict[c] += 1
        return word_dict

    word_dict = get_dict(train.append(testA))
    word_dict = {w: c for w, c in word_dict.items() if c >= min_count}
    word_dict = dict(sorted(word_dict.items(), key=lambda s: -s[1]))
    word_dict = list(word_dict.keys())
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "sim_no", 'sim', 'unlabeled']
    words = special_tokens + word_dict
    pd.Series(words).to_csv(f"{conf.DATA_PATH}/bert-vocab.txt", header=False, index=0)
    print(f"bert vocab saved at {f'{conf.DATA_PATH}/bert-vocab.txt'}")


def load_vocab_and_align_with_bert():
    vocab = pd.read_csv(f"{conf.DATA_PATH}/bert-vocab.txt", names=['words'])
    vocab_dict = {}
    for key, value in vocab.words.to_dict().items():
        vocab_dict[value] = key

    with open(f"{conf.DATA_ROOT}/bert_chinese_vocab.txt", encoding='utf-8') as fr:
        lines = fr.read()
        tokens = lines.split('\n')
    token_dict = dict(zip(tokens, range(len(tokens))))
    counts = json.load(open(f"{conf.DATA_ROOT}/counts.json"))
    del counts['[CLS]']
    del counts['[SEP]']
    freqs = [counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])]
    keep_tokens = list(np.argsort(freqs)[::-1])
    keep_tokens = [0, 100, 101, 102, 103, 5, 6, 7] + keep_tokens[:len(vocab_dict)]
    return vocab_dict, keep_tokens


class MlmNspDataset(Dataset):
    def __init__(self, data: pd.DataFrame, vocab: dict, seq_len: int = 128):
        self.lines = data
        self.vocab = vocab
        self.seq_len = seq_len
        self.size = self.lines.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        t1, t2, is_nsp = self._get_sentence(index)  # NSP 任务
        t1_rand, t1_label = self._random_word(t1)  # MLM 任务
        t2_rand, t2_label = self._random_word(t2)  # MLM 任务
        label = self.lines.label.iloc[index]

        t1 = [self.vocab['[CLS]']] + t1_rand + [self.vocab['[SEP]']]
        t2 = t2_rand + [self.vocab['[SEP]']]

        # label=[0, ], +5=> 不相似为5， 相似为6，未标注为7
        t1_label = [label + 5] + t1_label + [self.vocab['[PAD]']]
        t2_label = t2_label + [self.vocab['[PAD]']]

        token_type_ids = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        padding_label = [-100 for _ in range(self.seq_len - len(bert_input))]
        attention_mask = len(bert_input) * [1] + len(padding) * [0]
        bert_input.extend(padding)
        bert_label.extend(padding_label)
        token_type_ids.extend(padding)

        bert_input = np.asarray(bert_input)
        bert_label = np.asarray(bert_label)
        token_type_ids = np.asarray(token_type_ids)
        attention_mask = np.asarray(attention_mask)
        is_nsp = np.asarray(is_nsp)

        encoding = {
            'input_ids': bert_input,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'bert_label': bert_label
        }
        return encoding, is_nsp

    def _get_sentence(self, idx):
        t1, t2, _ = self.lines.iloc[idx].values
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            neg_idx = random.randrange(self.size)
            if neg_idx == idx:
                neg_idx = (neg_idx + 1) % self.size
            return t1, self.lines.iloc[neg_idx].values[1], 0

    def _random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:  # 80% [MASK]
                    tokens[i] = self.vocab['[MASK]']
                elif prob < 0.9:  # 10% random
                    tokens[i] = random.randrange(len(self.vocab))
                else:  # 10% self
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])

                output_label.append(self.vocab.get(token, self.vocab['[UNK]']))
            else:
                tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                output_label.append(-100)
        return tokens, output_label
