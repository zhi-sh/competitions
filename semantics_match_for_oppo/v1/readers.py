# -*- coding: utf-8 -*-
# @DateTime :2021/3/16
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import random
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from settings import DATA_PATH


def build_vocab(train, testA):
    def get_dict(data):
        word_dict = defaultdict(int)
        for i in tqdm(range(data.shape[0])):
            text = data.text_a.iloc[i].split() + data.text_b.iloc[i].split()
            for c in text:
                word_dict[c] += 1
        return word_dict

    # Main Process
    train_dict = get_dict(train)
    testA_dict = get_dict(testA)
    word_dict = list(train_dict.keys()) + list(testA_dict.keys())
    word_dict = set(word_dict)  # 去重 {'11', '22'}
    word_dict = set(map(int, word_dict))  # {11, 22}
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = special_tokens + list(word_dict)
    pd.Series(words).to_csv(f"{DATA_PATH}/bert-vocab.txt", header=False, index=False)
    vocab = pd.read_csv(f"{DATA_PATH}/bert-vocab.txt", names=['words'])
    vocab_dict = {}
    for key, value in vocab.words.to_dict().items():
        vocab_dict[value] = key
    return vocab_dict


def load_vocab():
    vocab = pd.read_csv(f"{DATA_PATH}/bert-vocab.txt", names=['words'])
    vocab_dict = {}
    for key, value in vocab.words.to_dict().items():
        vocab_dict[value] = key
    return vocab_dict


class MlmNspDataset(Dataset):
    def __init__(self, corpus_path: str, vocab: dict, seq_len: int = 128):
        self.corpus_path = corpus_path
        self.vocab = vocab
        self.seq_len = seq_len
        self.lines = pd.read_csv(corpus_path, sep='\t', names=['text_a', 'text_b', 'label'])
        self.size = self.lines.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        t1, t2, is_nsp = self._get_sentence(index)  # NSP 任务
        t1_rand, t1_label = self._random_word(t1)  # MLM 任务
        t2_rand, t2_label = self._random_word(t2)  # MLM 任务

        t1 = [self.vocab['[CLS]']] + t1_rand + [self.vocab['[SEP]']]
        t2 = t2_rand + [self.vocab['[SEP]']]
        t1_label = [self.vocab['[PAD]']] + t1_label + [self.vocab['[PAD]']]
        t2_label = t2_label + [self.vocab['[PAD]']]

        token_type_ids = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        attention_mask = len(bert_input) * [1] + len(padding) * [0]
        bert_input.extend(padding)
        bert_label.extend(padding)
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


class CustomDataset(Dataset):
    def __init__(self, data: pd.DataFrame, word_dict: dict, seq_len: int = 50):
        self.data = data
        self.vocab = word_dict
        self.seq_len = seq_len

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        text_a, text_b, label = self.data.iloc[index].values
        text_a = self._get_sentence(text_a)
        text_b = self._get_sentence(text_b)
        text_a = [self.vocab['[CLS]']] + text_a + [self.vocab['[SEP]']]
        text_b = text_b + [self.vocab['[SEP]']]

        token_type_ids = ([0 for _ in range(len(text_a))] + [1 for _ in range(len(text_b))])[:self.seq_len]
        text = (text_a + text_b)[:self.seq_len]
        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(text))]
        attention_mask = len(text) * [1]
        text.extend(padding)
        token_type_ids.extend(padding)
        attention_mask.extend(padding)
        text = np.asarray(text)
        token_type_ids = np.asarray(token_type_ids)
        attention_mask = np.asarray(attention_mask)

        encoding = {
            'input_ids': text,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        return encoding, label

    def _get_sentence(self, sentence):
        tokens = sentence.split()
        for i in range(len(tokens)):
            tokens[i] = self.vocab.get(tokens[i], self.vocab['[UNK]'])
        return tokens
